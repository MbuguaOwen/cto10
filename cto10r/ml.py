import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from tqdm.auto import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss


def _log(msg: str) -> None:
    print(msg, flush=True)


def wilson_lcb(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + (z*z)/n
    centre = p + (z*p*(1-p) + (z*z)/4.0) / (2*n)
    # Use standard formula for margin
    margin = z * math.sqrt((p*(1-p) + (z*z)/(4*n)) / n)
    return max(0.0, (centre - margin) / denom)


def auto_bin_train(series: pd.Series, n_bins: int):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(x) == 0 or n_bins < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(x, qs, method="linear")
    edges[0] = -np.inf; edges[-1] = np.inf
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-12
    return edges


def apply_bins(series: pd.Series, edges: np.ndarray) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    idx = np.clip(np.digitize(x, edges, right=False) - 1, 0, len(edges)-2)
    return pd.Series(idx.astype(np.int16), index=series.index)


def _safe_corrcoef(X: np.ndarray) -> np.ndarray:
    # X shape: [n_samples, n_features]
    if X.ndim != 2:
        cols = X.shape[1] if X.ndim > 1 else (X.shape[0] if X.ndim == 1 else 0)
        return np.eye(max(1, cols), dtype=float)
    if X.shape[0] == 0 or X.shape[1] < 2:
        return np.eye(max(1, X.shape[1]), dtype=float)
    col_std = np.nanstd(X, axis=0)
    keep = np.isfinite(col_std) & (col_std > 1e-12)
    if keep.sum() < 2:
        return np.eye(int(keep.sum()), dtype=float)
    Xk = X[:, keep]
    Xk = (Xk - np.nanmean(Xk, axis=0)) / np.nanstd(Xk, axis=0)
    C = np.corrcoef(np.nan_to_num(Xk), rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return C


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    if len(probs) == 0:
        return 0.0
    mask = np.isfinite(probs) & np.isfinite(y)
    if not mask.any():
        return 0.0
    probs = probs[mask]
    y = y[mask]
    order = np.argsort(probs)
    probs = probs[order]
    y = y[order]
    splits = np.linspace(0, len(y), n_bins + 1, dtype=int)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        a, b = splits[i], splits[i + 1]
        if b <= a:
            continue
        p_bin = probs[a:b]
        y_bin = y[a:b]
        if len(y_bin) == 0:
            continue
        e = abs(float(p_bin.mean()) - float(y_bin.mean())) * (len(y_bin) / max(1, n))
        ece += e
    return float(ece)


def _hist_by_month(s: pd.Series, edges: np.ndarray, months: pd.Series) -> Dict[str, np.ndarray]:
    res: Dict[str, np.ndarray] = {}
    if len(s) == 0:
        return res
    months = months.reindex(s.index)
    for m, sub in s.groupby(months, observed=False):
        sub = pd.to_numeric(sub, errors="coerce").dropna()
        if len(sub) == 0:
            continue
        idx = np.clip(np.digitize(sub.to_numpy(), edges, right=False) - 1, 0, len(edges) - 2)
        hist = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        if hist.sum() > 0:
            hist /= hist.sum()
        res[str(m)] = hist
    return res


def _max_js_across_months(hists: Dict[str, np.ndarray]) -> float:
    keys = list(hists.keys())
    if len(keys) < 2:
        return 0.0
    mx = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p = hists[keys[i]]
            q = hists[keys[j]]
            if p.sum() == 0 or q.sum() == 0:
                continue
            mx = max(mx, float(jensenshannon(p, q)))
    return mx


def _choose_bins_for_feature(
    df_fit: pd.DataFrame,
    df_val: pd.DataFrame,
    col: str,
    y_fit: np.ndarray,
    y_val: np.ndarray,
    months_fit: List[str],
    cfg: dict,
) -> List[float]:
    bin_cfg = cfg.get("binning", {})
    CANDS = [int(c) for c in bin_cfg.get("candidates", [3, 4, 5, 7, 10]) if int(c) >= 2]
    if not CANDS:
        CANDS = [3, 4, 5]
    MIN_FR = float(bin_cfg.get("min_bin_frac", 0.005))
    MI_DEL = float(bin_cfg.get("mi_delta_min", 0.002))
    ECE_DEL = float(bin_cfg.get("ece_delta_min", 0.002))
    MAX_JS = float(bin_cfg.get("max_js_month_drift", 0.15))
    PREF_MONO = bool(bin_cfg.get("prefer_monotone", True))

    if col not in df_fit.columns:
        return [-np.inf, np.inf]

    x_fit = pd.to_numeric(df_fit[col], errors="coerce")
    x_val = pd.to_numeric(df_val[col], errors="coerce")
    valid_fit = x_fit.dropna()
    if valid_fit.empty:
        return [-np.inf, np.inf]
    if valid_fit.nunique(dropna=True) <= 1:
        return [-np.inf, np.inf]

    logistic_ok = len(np.unique(y_fit)) > 1 and len(np.unique(y_val)) > 1
    best: Tuple[float, float, float, List[float]] | None = None
    prev_mi, prev_ece = 0.0, 1.0

    if "ym" in df_fit.columns:
        months_series = df_fit["ym"]
    elif months_fit:
        months_series = pd.Series(months_fit[0], index=df_fit.index)
    else:
        months_series = pd.Series("_", index=df_fit.index)

    for k in CANDS:
        try:
            qs = np.linspace(0, 1, k + 1)
            edges = np.quantile(valid_fit.to_numpy(), qs, method="linear")
        except Exception:
            continue
        edges = np.asarray(edges, dtype=float)
        if edges.size < 2:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9

        changed = True
        while changed and len(edges) > 2:
            changed = False
            idx = np.clip(np.digitize(x_fit, edges, right=False) - 1, 0, len(edges) - 2)
            fracs = np.bincount(idx, minlength=len(edges) - 1).astype(float)
            total = fracs.sum()
            if total <= 0:
                break
            fracs /= total
            small = np.where(fracs < MIN_FR)[0]
            if len(small):
                i = int(small[0])
                if i == len(edges) - 2:
                    merge_to = i - 1
                elif i == 0:
                    merge_to = 1
                else:
                    merge_to = i + 1 if fracs[i + 1] <= fracs[i - 1] else i - 1
                rm = min(i, merge_to) + 1
                if 0 < rm < len(edges) - 1:
                    edges = np.delete(edges, rm)
                    changed = True

        if len(edges) < 4:
            edges = np.quantile(x_fit.dropna(), [0, 1 / 3, 2 / 3, 1.0], method="linear")
            edges = np.asarray(edges, dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-9

        idx_fit = np.clip(np.digitize(x_fit, edges, right=False) - 1, 0, len(edges) - 2)
        idx_val = np.clip(np.digitize(x_val, edges, right=False) - 1, 0, len(edges) - 2)

        Xf = pd.get_dummies(idx_fit, drop_first=False)
        if Xf.empty:
            continue
        Xv = pd.get_dummies(idx_val, drop_first=False).reindex(columns=Xf.columns, fill_value=0)

        try:
            if logistic_ok:
                lr = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=200)
                lr.fit(Xf.values, y_fit)
                p_val = lr.predict_proba(Xv.values)[:, 1]
            else:
                p_val = np.repeat(float(np.mean(y_fit)), len(Xv))
        except Exception:
            continue

        try:
            arr = np.column_stack(
                [np.asarray(p_val, dtype=float), np.asarray(y_val, dtype=float)]
            )
            C = _safe_corrcoef(arr)
            if C.shape[0] >= 2 and C.shape[1] >= 2:
                mi_score = abs(float(C[0, 1]))
            else:
                mi_score = 0.0
        except Exception:
            mi_score = 0.0

        ece = _ece(np.asarray(p_val, dtype=float), np.asarray(y_val, dtype=float), n_bins=10)
        try:
            pr = float(average_precision_score(y_val, p_val))
        except Exception:
            pr = 0.0
        try:
            br = float(brier_score_loss(y_val, p_val))
        except Exception:
            br = 1.0

        hists = _hist_by_month(x_fit, edges, months_series)
        js = _max_js_across_months(hists)

        mono_ok = True
        if PREF_MONO and len(np.unique(idx_fit)) > 1:
            means = (
                pd.DataFrame({"bin": idx_fit, "y": y_fit})
                .groupby("bin", observed=False)["y"]
                .mean()
                .fillna(0.0)
                .to_numpy()
            )
            if len(means) > 1:
                mono_ok = bool(
                    np.all(np.diff(means) >= -0.02) or np.all(np.diff(means) <= 0.02)
                )

        imp_mi = (mi_score - prev_mi) >= MI_DEL
        imp_ece = (prev_ece - ece) >= ECE_DEL
        accept = (js <= MAX_JS) and mono_ok and (imp_mi or imp_ece)

        cand = (pr, -ece, -br, mi_score, edges.tolist())
        if (best is None and accept) or (accept and cand > best):
            best = cand
            prev_mi, prev_ece = mi_score, ece

    if best is None:
        fallback = auto_bin_train(x_fit, max(3, CANDS[0]))
        return fallback.tolist()
    return best[-1]


def select_feature_cols(df: pd.DataFrame, inc, exc):
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in inc) and not any(c.startswith(p) for p in exc):
            cols.append(c)
    return cols


COND_ALIASES: Dict[str, Tuple[str, int]] = {
    "high": ("top", 1),
    "high2": ("top", 2),
    "low": ("bot", 1),
    "low2": ("bot", 2),
    "old": ("top", 1),
    "near": ("bot", 1),
    "near_or_low": ("bot", 2),
    "pos": ("top", 1),
    "neg": ("bot", 1),
    "eq1": ("eq", 1),
}


def _resolve_curated_column(df_cat: pd.DataFrame, col: str) -> str | None:
    if col in df_cat.columns:
        return col
    alt = f"q_{col}"
    if alt in df_cat.columns:
        return alt
    return None


def _mask_for_condition(df_cat: pd.DataFrame, col: str, spec: str) -> pd.Series:
    kind, k = COND_ALIASES.get(spec, ("top", 1))
    if pd.api.types.is_categorical_dtype(df_cat[col]):
        vals = df_cat[col].astype("Int64")
    else:
        vals = pd.to_numeric(df_cat[col], errors="coerce")
    vals = vals.astype("float")
    if np.isnan(vals.to_numpy()).all():
        return pd.Series(False, index=df_cat.index)
    valid = vals.dropna()
    if valid.empty:
        return pd.Series(False, index=df_cat.index)
    vmax = float(valid.max())
    vmin = float(valid.min())
    if kind == "top":
        thr = vmax - (k - 1)
        mask = vals >= thr
    elif kind == "bot":
        thr = vmin + (k - 1)
        mask = vals <= thr
    elif kind == "eq":
        mask = vals == float(k)
    else:
        mask = vals >= vmax
    return mask.fillna(False)


def build_curated_crosses(df_cat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cr_cfg = cfg.get("crosses", {})
    if not cr_cfg.get("enable_curated", True):
        return pd.DataFrame(index=df_cat.index)
    pairs = cr_cfg.get("curated", [])
    feats = {}
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = pair
        ca, sa = a.split(":", 1) if ":" in a else (a, "high")
        cb, sb = b.split(":", 1) if ":" in b else (b, "high")
        col_a = _resolve_curated_column(df_cat, ca)
        col_b = _resolve_curated_column(df_cat, cb)
        if col_a is None or col_b is None:
            continue
        mask = _mask_for_condition(df_cat, col_a, sa) & _mask_for_condition(df_cat, col_b, sb)
        name = f"xC_{ca}:{sa}__{cb}:{sb}"
        feats[name] = mask.astype("int8")
    if not feats:
        return pd.DataFrame(index=df_cat.index)
    return pd.DataFrame(feats, index=df_cat.index)


def _limit_and_filter_crosses(
    df_fit_cross: pd.DataFrame,
    df_val_cross: pd.DataFrame,
    y_fit: np.ndarray,
    y_val: np.ndarray,
    max_total: int,
) -> List[str]:
    if df_fit_cross.empty or max_total <= 0:
        return []
    support = (df_fit_cross.sum(axis=0) / max(1, len(df_fit_cross))).fillna(0.0)
    keep = [c for c, s in support.items() if float(s) >= 0.005]
    if not keep:
        return []
    keep.sort(key=lambda c: support[c], reverse=True)
    keep = keep[: max_total * 2]

    logistic_ok = len(np.unique(y_fit)) > 1 and len(np.unique(y_val)) > 1
    scores: List[Tuple[float, str]] = []
    for col in keep:
        Xf = df_fit_cross[[col]].astype("float32").to_numpy()
        if Xf.sum() <= 0 or np.unique(Xf).size < 2:
            continue
        Xv = df_val_cross.reindex(columns=[col], fill_value=0).astype("float32").to_numpy()
        if Xv.size == 0:
            continue
        if logistic_ok:
            try:
                lr = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=200)
                lr.fit(Xf, y_fit)
                pv = lr.predict_proba(Xv)[:, 1]
                ap = float(average_precision_score(y_val, pv))
            except Exception:
                continue
        else:
            ap = float(support[col])
        scores.append((ap, col))

    if not scores:
        return []
    scores.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scores[:max_total]]


def build_feature_matrix(df: pd.DataFrame, cfg: dict, is_train: bool, bin_schema: dict | None):
    inc = list(cfg.get("features", {}).get("include_prefixes", ["qbin_", "qbin_agebin_", "since_", "occ_"]))
    exc = list(cfg.get("features", {}).get("exclude_prefixes", []))

    cols = select_feature_cols(df, inc, exc)
    cont_cols = [c for c in cols if c.startswith(("since_", "occ_"))]
    cat_cols = [c for c in cols if c not in cont_cols]

    auto = str(cfg.get("n_bins_cont", "auto")).lower() == "auto"
    schema = {"cont": {}}
    if bin_schema:
        schema["cont"].update(bin_schema.get("cont", {}))
    bin_schema = schema

    if is_train:
        if auto:
            if not bin_schema["cont"]:
                n_fixed = 5
                for c in cont_cols:
                    edges = auto_bin_train(df.loc[:, c], n_fixed)
                    bin_schema["cont"][c] = edges.tolist()
        else:
            n_fixed = int(cfg.get("n_bins_cont", 5))
            for c in cont_cols:
                edges = auto_bin_train(df.loc[:, c], n_fixed)
                bin_schema["cont"][c] = edges.tolist()
    else:
        if not auto:
            n_fixed = int(cfg.get("n_bins_cont", 5))
        else:
            n_fixed = 5
        for c in cont_cols:
            if c not in bin_schema["cont"]:
                edges = auto_bin_train(df.loc[:, c], n_fixed)
                bin_schema["cont"][c] = edges.tolist()

    Xc = df[cat_cols].copy() if cat_cols else pd.DataFrame(index=df.index)
    newcols = {}
    for c in cont_cols:
        edges = np.array(bin_schema["cont"].get(c, [-np.inf, np.inf]), dtype=float)
        newcols[f"q_{c}"] = apply_bins(df[c], edges)
    if newcols:
        new_df = pd.DataFrame(newcols, index=df.index)
        Xc = pd.concat([Xc, new_df], axis=1, copy=False)

    if Xc.empty:
        Xc = pd.DataFrame(index=df.index)
    else:
        for c in Xc.columns:
            Xc[c] = Xc[c].astype("Int64").astype("category")

    if is_train:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
        X = enc.fit_transform(Xc)
        return X, enc, bin_schema, Xc.columns.tolist()
    return Xc, None, bin_schema, Xc.columns.tolist()


def _align_input_columns(Xc: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    expected = list(expected_cols or [])
    aligned = Xc.copy()
    if not expected:
        return aligned.reindex(columns=expected)

    n = len(aligned.index)
    for col in expected:
        if col not in aligned.columns:
            fill = pd.Categorical(np.full(n, -1, dtype=int), categories=[-1])
            aligned[col] = pd.Series(fill, index=aligned.index)
        else:
            series = aligned[col]
            if pd.api.types.is_categorical_dtype(series):
                if -1 not in series.cat.categories:
                    series = series.cat.add_categories([-1])
                series = series.fillna(-1)
                aligned[col] = series.astype("category")
            else:
                vals = pd.to_numeric(series, errors="coerce").fillna(-1).astype(
                    np.int64, copy=False
                )
                vals_np = vals.to_numpy(dtype=np.int64, copy=False)
                cats = np.unique(np.append(vals_np, -1))
                aligned[col] = pd.Series(
                    pd.Categorical(vals_np, categories=cats), index=aligned.index
                )

    extras = [c for c in aligned.columns if c not in expected]
    if extras:
        aligned = aligned.drop(columns=extras)
    return aligned.reindex(columns=expected)


def _parse_winner_conds_for_crosses(conds: list[dict], df_cat_cols: list[str]):
    pairs = []
    simple = []
    for c in conds:
        feat = str(c.get("feat",""))
        val  = c.get("thr", None)
        if val is None:
            continue
        # Map miner literal 'B_{orig}' back to cat feature column used in ML matrix
        if feat.startswith("B_"):
            base = feat[2:]
        else:
            base = feat
        if base.startswith("qbin_"):
            col = base
        else:
            col = f"q_{base}"
        if col in df_cat_cols:
            try:
                simple.append((col, int(val)))
            except Exception:
                continue
    # build pairwise from simple
    for i in range(len(simple)):
        for j in range(i+1, len(simple)):
            pairs.append((simple[i][0], simple[i][1], simple[j][0], simple[j][1]))
    return pairs


def build_motif_crosses(df_cat: pd.DataFrame, gating_path: Path, limit_pairs: int = 200):
    if not gating_path.exists():
        return pd.DataFrame(index=df_cat.index), []
    try:
        g = json.loads(gating_path.read_text(encoding="utf-8"))
        winners = g.get("promoted") or g.get("winners") or []
    except Exception:
        winners = []
    all_pairs = []
    for w in winners:
        conds = w.get("conds", [])
        pairs = _parse_winner_conds_for_crosses(conds, list(df_cat.columns))
        for p in pairs:
            all_pairs.append(p)
            if len(all_pairs) >= limit_pairs:
                break
        if len(all_pairs) >= limit_pairs:
            break
    crosses = {}
    used = []
    for (ci, vi, cj, vj) in all_pairs:
        feat = f"x_{ci}={vi}__{cj}={vj}"
        crosses[feat] = ((df_cat[ci].astype("Int64") == vi) & (df_cat[cj].astype("Int64") == vj)).astype("int8")
        used.append(feat)
    if not crosses:
        return pd.DataFrame(index=df_cat.index), []
    return pd.DataFrame(crosses, index=df_cat.index), used


def split_train_valid_by_month(df: pd.DataFrame, train_months: list[str]):
    if not train_months:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()
    val_m = sorted(train_months)[-1]
    df_fit = df[df["ym"].isin(train_months[:-1])].copy()
    df_val = df[df["ym"] == val_m].copy()
    if len(df_fit) == 0:
        df_fit = df.copy()
    if len(df_val) == 0:
        df_val = df.copy()
    return df_fit, df_val


def train_ml_gating(cands_tr: pd.DataFrame, events_tr: pd.DataFrame, train_months: list[str], cfg: dict, out_dir: Path):
    ml_cfg = cfg.get("ml_gating", {})
    if not ml_cfg.get("enabled", True):
        return None

    art_dir = Path(out_dir) / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    y = events_tr[["ts","outcome"]].copy()
    # exclude timeouts: keep only resolved outcomes
    y = y[y["outcome"].isin(["win","loss"])].copy()
    y["y"] = (y["outcome"] == "win").astype(int)
    df = cands_tr.merge(y[["ts","y"]], on="ts", how="inner")

    df_fit, df_val = split_train_valid_by_month(df, train_months)
    y_fit = df_fit["y"].to_numpy()
    y_val = df_val["y"].to_numpy()

    inc = list(ml_cfg.get("features", {}).get("include_prefixes", ["qbin_", "qbin_agebin_", "since_", "occ_"]))
    exc = list(ml_cfg.get("features", {}).get("exclude_prefixes", []))
    all_cols = select_feature_cols(df, inc, exc)
    cont_cols = [c for c in all_cols if c.startswith(("since_", "occ_"))]

    auto_bins = str(ml_cfg.get("n_bins_cont", "auto")).lower() == "auto"
    bin_schema = {"cont": {}}
    adaptive_logs: List[str] = []
    if auto_bins:
        for c in cont_cols:
            if c not in df_fit.columns:
                continue
            try:
                edges = _choose_bins_for_feature(df_fit, df_val, c, y_fit, y_val, train_months, ml_cfg)
            except Exception:
                edges = auto_bin_train(df_fit.loc[:, c], 5).tolist()
            if isinstance(edges, np.ndarray):
                edges = edges.tolist()
            bin_schema["cont"][c] = edges
            if len(edges) - 1 != 5:
                adaptive_logs.append(f"{c}={len(edges)-1}")
        if adaptive_logs:
            print(f"[ml] adaptive bins (auto): {', '.join(adaptive_logs)}", flush=True)
    else:
        n_fixed = int(ml_cfg.get("n_bins_cont", 5))
        for c in cont_cols:
            if c not in df_fit.columns:
                continue
            edges = auto_bin_train(df_fit.loc[:, c], n_fixed)
            bin_schema["cont"][c] = edges.tolist()

    X_fit_skel, enc, bin_schema, cols = build_feature_matrix(df_fit, ml_cfg, is_train=True, bin_schema=bin_schema)
    with open(art_dir / "ml_input_cols.json", "w", encoding="utf-8") as f:
        json.dump(list(cols), f, ensure_ascii=False, indent=2)
    X_fit_cat, _, _, _ = build_feature_matrix(df_fit, ml_cfg, is_train=False, bin_schema=bin_schema)
    X_val_cat, _, _, _ = build_feature_matrix(df_val, ml_cfg, is_train=False, bin_schema=bin_schema)
    expected_inputs = list(getattr(enc, "feature_names_in_", []))
    if not expected_inputs:
        expected_inputs = list(cols)
    X_fit_cat = _align_input_columns(X_fit_cat, expected_inputs)
    X_val_cat = _align_input_columns(X_val_cat, expected_inputs)
    X_val_base = enc.transform(X_val_cat)

    cross_cfg = ml_cfg.get("crosses", {})
    use_motif_crosses = bool(cross_cfg.get("use_motif_crosses", ml_cfg.get("use_motif_crosses", True)))
    enable_curated = bool(cross_cfg.get("enable_curated", True))
    cross_cap = int(cross_cfg.get("max_total", 30))
    motif_limit = int(cross_cfg.get("top_k_literals_from_winners", 20))

    df_train_cross = pd.DataFrame(index=df_fit.index)
    df_val_cross = pd.DataFrame(index=df_val.index)
    cross_sources: Dict[str, str] = {}
    gating_path = art_dir / "gating.json"

    motif_total = 0
    curated_total = 0
    if use_motif_crosses:
        xfit_motif, _ = build_motif_crosses(X_fit_cat, gating_path, limit_pairs=motif_limit)
        motif_total = xfit_motif.shape[1]
        if motif_total:
            xval_motif, _ = build_motif_crosses(X_val_cat, gating_path, limit_pairs=motif_limit)
            xval_motif = xval_motif.reindex(columns=xfit_motif.columns, fill_value=0)
            df_train_cross = pd.concat([df_train_cross, xfit_motif], axis=1)
            df_val_cross = pd.concat([df_val_cross, xval_motif], axis=1)
            for name in xfit_motif.columns:
                cross_sources[name] = "motif"

    xfit_cur = build_curated_crosses(X_fit_cat, ml_cfg) if enable_curated else pd.DataFrame(index=df_fit.index)
    curated_total = xfit_cur.shape[1]
    if curated_total:
        xval_cur = build_curated_crosses(X_val_cat, ml_cfg)
        xval_cur = xval_cur.reindex(columns=xfit_cur.columns, fill_value=0)
        df_train_cross = pd.concat([df_train_cross, xfit_cur], axis=1)
        df_val_cross = pd.concat([df_val_cross, xval_cur], axis=1)
        for name in xfit_cur.columns:
            cross_sources[name] = "curated"

    keep_names = _limit_and_filter_crosses(df_train_cross, df_val_cross, y_fit, y_val, cross_cap)
    motif_kept = [c for c in keep_names if cross_sources.get(c) == "motif"]
    curated_kept = [c for c in keep_names if cross_sources.get(c) == "curated"]
    if enable_curated:
        examples_cur = ", ".join(curated_kept[:3]) if curated_kept else "-"
        print(
            f"[ml] curated crosses kept {len(curated_kept)} of {curated_total} (cap {cross_cap}); examples: {examples_cur}",
            flush=True,
        )
    if use_motif_crosses:
        print(
            f"[ml] motif crosses kept {len(motif_kept)} of {motif_total} (cap {cross_cap})",
            flush=True,
        )
    print(f"[ml] total crosses used: {len(keep_names)} (cap {cross_cap})", flush=True)

    used_crosses = {"motif": motif_kept, "curated": curated_kept}
    with open(art_dir / "ml_used_crosses.json", "w", encoding="utf-8") as f:
        json.dump(used_crosses, f, ensure_ascii=False, indent=2)

    from scipy import sparse

    X_fit = X_fit_skel
    X_val = X_val_base
    if keep_names:
        train_cross = df_train_cross[keep_names].astype(np.float32)
        val_cross = df_val_cross.reindex(columns=keep_names, fill_value=0).astype(np.float32)
        X_fit = sparse.hstack([X_fit_skel, sparse.csr_matrix(train_cross.values)], format="csr")
        X_val = sparse.hstack([X_val_base, sparse.csr_matrix(val_cross.values)], format="csr")

    best = None
    model_name = (ml_cfg.get("model") or "logreg").lower()
    if model_name == "gbdt":
        n_pos = max(1, int(y_fit.sum()))
        n_neg = max(1, int((1 - y_fit).sum()))
        w_pos = (n_neg / (n_pos + n_neg))
        w_neg = (n_pos / (n_pos + n_neg))
        sw_fit = np.where(y_fit == 1, w_pos, w_neg)
        grid = ml_cfg.get("gbdt_grid", {"n_estimators":[200], "max_depth":[3], "learning_rate":[0.1], "subsample":[1.0]})
        for ne in grid.get("n_estimators", [200]):
            for md in grid.get("max_depth", [3]):
                for lr in grid.get("learning_rate", [0.1]):
                    for ss in grid.get("subsample", [1.0]):
                        clf = GradientBoostingClassifier(
                            n_estimators=int(ne), max_depth=int(md), learning_rate=float(lr), subsample=float(ss)
                        )
                        clf.fit(X_fit.toarray(), y_fit, sample_weight=sw_fit)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
                            cal.fit(X_val.toarray(), y_val)
                        p_val = cal.predict_proba(X_val.toarray())[:,1]
                        ap = average_precision_score(y_val, p_val)
                        best = max(best or (-1,None,None,{}), (ap, clf, cal, {"n_estimators":ne,"max_depth":md,"learning_rate":lr,"subsample":ss}), key=lambda t:t[0])
    else:
        c_grid = [float(c) for c in ml_cfg.get("c_grid", [0.1,0.2,0.5,1.0])]
        cw = ml_cfg.get("class_weight", "balanced")
        for C in c_grid:
            clf = LogisticRegression(penalty="l2", C=float(C), solver="liblinear", class_weight=cw, max_iter=300)
            clf.fit(X_fit, y_fit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
                cal.fit(X_val, y_val)
            p_val = cal.predict_proba(X_val)[:,1]
            ap = average_precision_score(y_val, p_val)
            best = max(best or (-1,None,None,{"C":C}), (ap, clf, cal, {"C":C}), key=lambda t:t[0])

    ap, clf, cal, params = best

    target_ppv = float(ml_cfg.get("target_ppv", 0.70))
    z = 1.96
    p_val = cal.predict_proba(X_val.toarray() if model_name == "gbdt" else X_val)[:, 1]
    dfv = pd.DataFrame({"p": p_val, "y": y_val})
    dfv = dfv.sort_values("p", ascending=False).reset_index(drop=True)
    dfv["tp"] = dfv["y"].cumsum()
    dfv["n"] = np.arange(1, len(dfv) + 1, dtype=int)
    dfv["ppv"] = dfv["tp"] / dfv["n"].clip(lower=1)
    dfv["lcb"] = dfv.apply(lambda r: wilson_lcb(int(r.tp), int(r.n), z), axis=1)
    total = len(dfv)
    if total > 0:
        ok = dfv[dfv["lcb"] >= target_ppv]
        if len(ok):
            k = int(ok["n"].iloc[0])
            k = max(1, min(k, total))
            tau = float(dfv.iloc[k - 1]["p"])
            cov = float(k / total)
            ach = float(dfv.iloc[k - 1]["lcb"])
        else:
            min_cov = float(ml_cfg.get("min_coverage_frac", 0.02))
            k = max(1, int(min_cov * total))
            k = max(1, min(k, total))
            tau = float(dfv.iloc[k - 1]["p"])
            cov = float(k / total)
            ach = float(dfv.iloc[k - 1]["lcb"])
    else:
        tau = 0.0
        cov = 0.0
        ach = 0.0

    sym = cfg.get("symbol") or cfg.get("sym") or ml_cfg.get("symbol") or ml_cfg.get("sym")
    if not sym:
        try:
            sym = Path(out_dir).parent.name
        except Exception:
            sym = "ml"
    sym = str(sym or "ml")

    _log(
        f"[{sym}] ml_gating: model={model_name} params={params} AP_val={ap:.3f} "
        f"tau={tau:.4f} target={target_ppv:.2f} LCB@tau={ach:.3f} cov={cov:.3f}"
    )

    import joblib
    joblib.dump(clf, art_dir / f"ml_{model_name}_base.pkl")
    joblib.dump(cal, art_dir / f"ml_{model_name}_calibrated.pkl")
    joblib.dump(enc, art_dir / "ml_encoder.pkl")
    with open(art_dir / "ml_bins.json","w",encoding="utf-8") as f: json.dump(bin_schema, f, ensure_ascii=False, indent=2)
    bin_sample: Dict[str, List[float]] = {}
    for idx, (feat, edges) in enumerate(bin_schema.get("cont", {}).items()):
        if idx >= 5:
            break
        try:
            seq = list(edges)
        except TypeError:
            seq = [edges]
        trimmed = seq[: min(len(seq), 6)] if seq else []
        bin_sample[feat] = [float(e) for e in trimmed]
    with open(art_dir / "ml_meta.json","w",encoding="utf-8") as f:
        json.dump({
            "model": model_name, "params": params, "ap_val": float(ap),
            "tau": tau, "target_ppv": target_ppv, "achieved_lcb": ach, "coverage_val": cov,
            "train_months": train_months, "used_crosses": used_crosses, "bin_sample": bin_sample
        }, f, ensure_ascii=False, indent=2)
    return {"tau": tau, "model": model_name}


def score_ml(cands_df: pd.DataFrame, fold_dir: Path, ml_cfg: dict) -> pd.Series:
    import joblib, json
    art = Path(fold_dir) / "artifacts"
    with open(art / "ml_meta.json","r",encoding="utf-8") as f: meta = json.load(f)
    model = meta.get("model","logreg")
    enc   = joblib.load(art / "ml_encoder.pkl")
    cal   = joblib.load(art / f"ml_{model}_calibrated.pkl")
    with open(art / "ml_bins.json","r",encoding="utf-8") as f: bins = json.load(f)

    Xc, _, _, _ = build_feature_matrix(cands_df, ml_cfg, is_train=False, bin_schema=bins)
    try:
        with open(art / "ml_input_cols.json", "r", encoding="utf-8") as f:
            persisted_cols = json.load(f)
    except Exception:
        persisted_cols = []
    encoder_inputs = list(getattr(enc, "feature_names_in_", []))
    expected_cols = encoder_inputs or list(persisted_cols)
    Xc = _align_input_columns(Xc, expected_cols)
    X = enc.transform(Xc)

    used_cross_path = art / "ml_used_crosses.json"
    if used_cross_path.exists():
        try:
            with open(used_cross_path, "r", encoding="utf-8") as f:
                used_cross_meta = json.load(f)
        except Exception:
            used_cross_meta = meta.get("used_crosses", {})
    else:
        used_cross_meta = meta.get("used_crosses", {})
    if isinstance(used_cross_meta, dict):
        motif_names = list(used_cross_meta.get("motif", []))
        curated_names = list(used_cross_meta.get("curated", []))
    else:
        motif_names = list(used_cross_meta)
        curated_names = []

    cross_cfg = ml_cfg.get("crosses", {})
    use_motif_crosses = bool(cross_cfg.get("use_motif_crosses", ml_cfg.get("use_motif_crosses", True))) or bool(motif_names)
    enable_curated = bool(cross_cfg.get("enable_curated", True)) or bool(curated_names)
    motif_limit = int(cross_cfg.get("top_k_literals_from_winners", 20))

    from scipy import sparse

    extra_blocks = []
    cross_counts = {"motif": 0, "curated": 0}
    if encoder_inputs:
        aligned = list(encoder_inputs) == list(Xc.columns)
        expected_len = len(encoder_inputs)
    else:
        aligned = list(expected_cols) == list(Xc.columns)
        expected_len = len(expected_cols)
    print(
        f"[ml] scoring: encoder columns aligned={aligned} (expected={expected_len}, got={len(Xc.columns)})",
        flush=True,
    )

    gating_path = art / "gating.json"
    if motif_names and use_motif_crosses:
        limit_pairs = max(len(motif_names), motif_limit)
        motif_df, _ = build_motif_crosses(Xc, gating_path, limit_pairs=limit_pairs)
        motif_df = motif_df.reindex(columns=motif_names, fill_value=0)
        if motif_df.shape[1]:
            extra_blocks.append(sparse.csr_matrix(motif_df.astype(np.float32).values))
            cross_counts["motif"] = motif_df.shape[1]

    if curated_names and enable_curated:
        curated_df = build_curated_crosses(Xc, ml_cfg)
        curated_df = curated_df.reindex(columns=curated_names, fill_value=0)
        if curated_df.shape[1]:
            extra_blocks.append(sparse.csr_matrix(curated_df.astype(np.float32).values))
            cross_counts["curated"] = curated_df.shape[1]

    if extra_blocks:
        X = sparse.hstack([X] + extra_blocks, format="csr")
    total_cross = sum(cross_counts.values())
    print(
        f"[ml] scoring: using crosses total={total_cross} (motif={cross_counts['motif']}, curated={cross_counts['curated']})",
        flush=True,
    )

    p = cal.predict_proba(X.toarray() if model=="gbdt" else X)[:,1]
    return pd.Series(p, index=cands_df.index, name="pwin")


def schedule_non_overlapping(
    df: pd.DataFrame,
    weight_mode: str = "expR",
    r_mult: float = 5.0,
    show_progress: bool = True,
    progress_desc: str = "schedule",
    update_every: int = 1000,
    timeout_sec: Optional[float] = None,
) -> pd.DataFrame:
    """
    Weighted non-overlapping interval selection (interval scheduling):
      - If weight_mode == "expR": weight = r_mult*pwin - (1-pwin)
      - If weight_mode in {"pwin","score"} and such column exists: use that column directly
      - Otherwise, weights default to 1.0 (uniform)
    Progress bar uses true, bounded totals; timeout triggers a deterministic greedy fallback.
    Always returns a scheduled subset of the *gated* input df.
    """
    if len(df) == 0:
        return df.copy()

    # require ts and end_ts/outcome_ts
    if "ts" not in df.columns:
        return df.copy()
    if "end_ts" in df.columns:
        end_ts = pd.to_numeric(df["end_ts"], errors="coerce").to_numpy()
    elif "outcome_ts" in df.columns:
        end_ts = pd.to_numeric(df["outcome_ts"], errors="coerce").to_numpy()
    else:
        return df.copy()

    ts = pd.to_numeric(df["ts"], errors="coerce").to_numpy()
    order = np.argsort(end_ts, kind="mergesort")
    ts = ts[order]
    end_ts = end_ts[order]

    # compute weights
    if weight_mode in ("pwin", "score"):
        col = "pwin" if "pwin" in df.columns else ("score" if "score" in df.columns else None)
        if col is not None:
            w = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()[order]
        else:
            w = np.ones_like(ts, dtype=float)
    elif weight_mode == "expR":
        if "pwin" in df.columns:
            p = pd.to_numeric(df["pwin"], errors="coerce").fillna(0.0).to_numpy()[order]
        else:
            p = np.ones_like(ts, dtype=float)
        w = (float(r_mult) * p) - (1.0 - p)
    else:
        w = np.ones_like(ts, dtype=float)

    # rightmost compatible index for each i: end_ts[j] <= ts[i]
    import bisect

    pidx = [bisect.bisect_right(end_ts, ts[i]) - 1 for i in range(len(ts))]
    n = len(ts)
    if update_every is None or update_every <= 0:
        update_every = 1000

    start = time.time()

    try:
        # DP forward
        dp = np.zeros(n + 1, dtype=float)
        choose = np.zeros(n, dtype=bool)
        for i in range(1, n + 1):
            skip = dp[i - 1]
            take = w[i - 1] + (dp[pidx[i - 1] + 1] if pidx[i - 1] >= 0 else 0.0)
            if take > skip:
                dp[i] = take
                choose[i - 1] = True
            else:
                dp[i] = skip
            if timeout_sec and (time.time() - start) > timeout_sec:
                raise TimeoutError("schedule timeout")

        # Backtrack with bounded progress
        pbar = tqdm(total=n, desc=progress_desc, disable=not show_progress)
        last_tick = 0
        sel = []
        i = n
        while i > 0:
            if choose[i - 1]:
                sel.append(i - 1)
                i = pidx[i - 1] + 1
            else:
                i -= 1
            if show_progress:
                last_tick += 1
                if last_tick >= update_every:
                    pbar.update(last_tick)
                    last_tick = 0
        if show_progress:
            if last_tick:
                pbar.update(last_tick)
            pbar.close()

        sel = np.array(sorted(sel))
        return df.iloc[order[sel]].copy()

    except (KeyboardInterrupt, TimeoutError, MemoryError):
        # Greedy deterministic fallback by (weight desc, end_ts asc) with accurate progress
        # IMPORTANT: np.lexsort uses the LAST key as primary sort key.
        idx = np.lexsort((end_ts, -w))
        end_ts_sorted = end_ts[idx]
        ts_sorted = ts[idx]

        keep = np.zeros(n, dtype=bool)
        last_end = -np.inf
        pbar = tqdm(total=n, desc=f"{progress_desc}[greedy]", disable=not show_progress)
        tick = 0
        for k in range(n):
            i = idx[k]
            if ts_sorted[k] >= last_end:
                keep[i] = True
                last_end = end_ts_sorted[k]
            tick += 1
            if show_progress and (tick % update_every == 0):
                pbar.update(update_every)
        if show_progress:
            if tick % update_every:
                pbar.update(tick % update_every)
            pbar.close()

        sel = np.flatnonzero(keep)
        return df.iloc[order[sel]].copy()


def _extract_y(df: pd.DataFrame) -> np.ndarray:
    if "y" not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float, copy=False)
    return y


def build_design_matrix(
    df: pd.DataFrame,
    features_cfg: dict,
    crosses_cap: int = 30,
    val_df: pd.DataFrame | None = None,
):
    """Fit feature transforms on df and return (X, y, feature_meta)."""
    from scipy import sparse

    work = df.copy()
    y = _extract_y(work)

    ml_cfg = features_cfg or {}
    inc = list(ml_cfg.get("features", {}).get("include_prefixes", ["qbin_", "qbin_agebin_", "since_", "occ_"]))
    exc = list(ml_cfg.get("features", {}).get("exclude_prefixes", []))
    all_cols = select_feature_cols(work, inc, exc)
    cont_cols = [c for c in all_cols if c.startswith(("since_", "occ_"))]

    bin_schema = {"cont": {}}
    auto_bins = str(ml_cfg.get("n_bins_cont", "auto")).lower() == "auto"

    if val_df is not None and not val_df.empty:
        val_ref = val_df.copy()
    else:
        val_ref = work.copy()
    y_val = _extract_y(val_ref)
    months_fit = sorted(map(str, work.get("ym", pd.Series([], dtype=str)).dropna().unique().tolist()))

    if auto_bins:
        for c in cont_cols:
            if c not in work.columns:
                continue
            try:
                edges = _choose_bins_for_feature(work, val_ref, c, y, y_val, months_fit, ml_cfg)
            except Exception:
                edges = auto_bin_train(work.loc[:, c], 5)
            if isinstance(edges, np.ndarray):
                edges = edges.tolist()
            bin_schema["cont"][c] = edges
    else:
        n_fixed = int(ml_cfg.get("n_bins_cont", 5))
        for c in cont_cols:
            if c not in work.columns:
                continue
            edges = auto_bin_train(work.loc[:, c], n_fixed)
            if isinstance(edges, np.ndarray):
                edges = edges.tolist()
            bin_schema["cont"][c] = edges

    X_train_skel, enc, bin_schema, cat_cols = build_feature_matrix(work, ml_cfg, is_train=True, bin_schema=bin_schema)
    expected_inputs = list(getattr(enc, "feature_names_in_", [])) or list(cat_cols)

    X_cat, _, _, _ = build_feature_matrix(work, ml_cfg, is_train=False, bin_schema=bin_schema)
    X_cat = _align_input_columns(X_cat, expected_inputs)
    X_base = enc.transform(X_cat)

    if val_ref is not None and len(val_ref):
        X_val_cat, _, _, _ = build_feature_matrix(val_ref, ml_cfg, is_train=False, bin_schema=bin_schema)
        X_val_cat = _align_input_columns(X_val_cat, expected_inputs)
    else:
        X_val_cat = None

    cross_cfg = ml_cfg.get("crosses", {})
    use_motif = bool(cross_cfg.get("use_motif_crosses", ml_cfg.get("use_motif_crosses", True)))
    enable_curated = bool(cross_cfg.get("enable_curated", True))
    motif_limit = int(cross_cfg.get("top_k_literals_from_winners", 20))
    gating_path = ml_cfg.get("gating_path")
    gating_path = Path(gating_path) if gating_path else None

    df_train_cross = pd.DataFrame(index=work.index)
    df_val_cross = pd.DataFrame(index=X_val_cat.index) if X_val_cat is not None else pd.DataFrame()
    cross_sources: Dict[str, str] = {}

    if crosses_cap > 0:
        if use_motif and gating_path and gating_path.exists():
            motif_train, _ = build_motif_crosses(X_cat, gating_path, limit_pairs=motif_limit)
            if motif_train.shape[1]:
                df_train_cross = pd.concat([df_train_cross, motif_train], axis=1)
                cross_sources.update({c: "motif" for c in motif_train.columns})
                if X_val_cat is not None:
                    motif_val, _ = build_motif_crosses(X_val_cat, gating_path, limit_pairs=motif_limit)
                    motif_val = motif_val.reindex(columns=motif_train.columns, fill_value=0)
                    df_val_cross = pd.concat([df_val_cross, motif_val], axis=1)
        if enable_curated:
            curated_train = build_curated_crosses(X_cat, ml_cfg)
            if curated_train.shape[1]:
                df_train_cross = pd.concat([df_train_cross, curated_train], axis=1)
                cross_sources.update({c: "curated" for c in curated_train.columns})
                if X_val_cat is not None:
                    curated_val = build_curated_crosses(X_val_cat, ml_cfg)
                    curated_val = curated_val.reindex(columns=curated_train.columns, fill_value=0)
                    df_val_cross = pd.concat([df_val_cross, curated_val], axis=1)

    if df_train_cross.empty or crosses_cap <= 0:
        keep_names: List[str] = []
    else:
        if df_val_cross.empty:
            df_val_cross = df_train_cross.copy()
            y_val_use = y
        else:
            y_val_use = y_val
        keep_names = _limit_and_filter_crosses(
            df_train_cross,
            df_val_cross,
            y,
            y_val_use,
            int(crosses_cap),
        )

    X = X_base
    if keep_names:
        train_cross = df_train_cross[keep_names].astype(np.float32)
        X = sparse.hstack([X_base, sparse.csr_matrix(train_cross.values)], format="csr")

    feature_meta = {
        "encoder": enc,
        "bin_schema": bin_schema,
        "expected_inputs": expected_inputs,
        "cross_names": keep_names,
        "cross_sources": {name: cross_sources.get(name, "") for name in keep_names},
        "gating_path": str(gating_path) if gating_path else None,
        "motif_limit": motif_limit,
        "literal_columns": list(X_cat.columns),
    }

    return X, y, feature_meta


def transform_design_matrix(
    df: pd.DataFrame,
    features_cfg: dict,
    feature_meta: dict,
):
    """Apply previously fitted feature_meta to df."""
    from scipy import sparse

    work = df.copy()
    y = _extract_y(work)
    bin_schema = feature_meta.get("bin_schema", {"cont": {}})
    enc = feature_meta["encoder"]
    expected_inputs = feature_meta.get("expected_inputs", [])

    X_cat, _, _, _ = build_feature_matrix(work, features_cfg, is_train=False, bin_schema=bin_schema)
    X_cat = _align_input_columns(X_cat, expected_inputs)
    X_base = enc.transform(X_cat)

    literal_df = X_cat.apply(lambda col: col.astype(str))

    blocks = [X_base]
    cross_names = feature_meta.get("cross_names", [])
    cross_sources = feature_meta.get("cross_sources", {})
    gating_path = feature_meta.get("gating_path")
    motif_limit = int(feature_meta.get("motif_limit", len(cross_names)))

    if cross_names:
        motif_names = [name for name in cross_names if cross_sources.get(name) == "motif"]
        curated_names = [name for name in cross_names if cross_sources.get(name) == "curated"]
        if motif_names and gating_path:
            gp = Path(gating_path)
            motif_df, _ = build_motif_crosses(X_cat, gp, limit_pairs=max(len(motif_names), motif_limit))
            motif_df = motif_df.reindex(columns=motif_names, fill_value=0)
            if motif_df.shape[1]:
                blocks.append(sparse.csr_matrix(motif_df.astype(np.float32).values))
        if curated_names:
            curated_df = build_curated_crosses(X_cat, features_cfg)
            curated_df = curated_df.reindex(columns=curated_names, fill_value=0)
            if curated_df.shape[1]:
                blocks.append(sparse.csr_matrix(curated_df.astype(np.float32).values))

    if len(blocks) == 1:
        X = blocks[0]
    else:
        X = sparse.hstack(blocks, format="csr")

    return X, y, literal_df


def train_classifier(
    X,
    y: np.ndarray,
    features_cfg: dict,
    X_val=None,
    y_val: np.ndarray | None = None,
):
    model_name = str(features_cfg.get("model", "logreg")).lower()
    y_fit = np.asarray(y, dtype=float)
    mask = np.isfinite(y_fit)
    if mask.any():
        X_fit = X[mask]
        y_fit = y_fit[mask]
    else:
        X_fit = X
    y_fit = y_fit.astype(int)

    if y_val is not None:
        y_val = np.asarray(y_val, dtype=float)

    if model_name == "gbdt":
        from scipy import sparse

        X_dense = X_fit.toarray() if sparse.issparse(X_fit) else np.asarray(X_fit)
        class_weight = str(features_cfg.get("class_weight", "balanced"))
        if class_weight == "balanced":
            pos = max(1, int((y_fit == 1).sum()))
            neg = max(1, int((y_fit == 0).sum()))
            w_pos = neg / (pos + neg)
            w_neg = pos / (pos + neg)
            sample_weight = np.where(y_fit == 1, w_pos, w_neg)
        else:
            sample_weight = None
        grid = features_cfg.get(
            "gbdt_grid",
            {"n_estimators": [200], "max_depth": [3], "learning_rate": [0.1], "subsample": [1.0]},
        )
        best: Tuple[float, GradientBoostingClassifier] | None = None
        for ne in grid.get("n_estimators", [200]):
            for md in grid.get("max_depth", [3]):
                for lr in grid.get("learning_rate", [0.1]):
                    for ss in grid.get("subsample", [1.0]):
                        clf = GradientBoostingClassifier(
                            n_estimators=int(ne),
                            max_depth=int(md),
                            learning_rate=float(lr),
                            subsample=float(ss),
                        )
                        clf.fit(X_dense, y_fit, sample_weight=sample_weight)
                        if X_val is not None and y_val is not None and len(y_val):
                            X_val_dense = X_val.toarray() if sparse.issparse(X_val) else np.asarray(X_val)
                            mask_val = np.isfinite(y_val)
                            if mask_val.any():
                                y_eval = y_val[mask_val].astype(int)
                                X_eval = X_val_dense[mask_val]
                            else:
                                y_eval = y_fit
                                X_eval = X_dense
                        else:
                            y_eval = y_fit
                            X_eval = X_dense
                        p = clf.predict_proba(X_eval)[:, 1]
                        score = average_precision_score(y_eval, p)
                        if best is None or score > best[0]:
                            best = (score, clf)
        if best is None:
            raise RuntimeError("GBDT training failed to produce a model")
        return best[1]

    c_grid = [float(c) for c in features_cfg.get("c_grid", [0.1, 0.2, 0.5, 1.0])]
    class_weight = features_cfg.get("class_weight", "balanced")
    best_lr: Tuple[float, LogisticRegression] | None = None
    for C in c_grid:
        clf = LogisticRegression(
            penalty="l2",
            C=float(C),
            solver="liblinear",
            class_weight=class_weight,
            max_iter=400,
        )
        clf.fit(X_fit, y_fit)
        if X_val is not None and y_val is not None and len(y_val):
            mask_val = np.isfinite(y_val)
            if mask_val.any():
                y_eval = y_val[mask_val].astype(int)
                X_eval = X_val[mask_val]
            else:
                y_eval = y_fit
                X_eval = X_fit
        else:
            y_eval = y_fit
            X_eval = X_fit
        p = clf.predict_proba(X_eval)[:, 1]
        score = average_precision_score(y_eval, p)
        if best_lr is None or score > best_lr[0]:
            best_lr = (score, clf)

    if best_lr is None:
        raise RuntimeError("Logistic regression training failed to produce a model")
    return best_lr[1]


def calibrate_probabilities(clf, X, y, method: str = "isotonic"):
    method_name = str(method).lower()
    cal_method = "sigmoid" if method_name in {"platt", "sigmoid"} else "isotonic"

    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(y_arr)
    if mask.any():
        X_fit = X[mask]
        y_fit = y_arr[mask].astype(int)
    else:
        X_fit = X
        y_fit = y_arr.astype(int)

    unique, counts = np.unique(y_fit, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 0
    n_samples = int(len(y_fit))

    if min_class < 2 or n_samples < 2:
        # Degenerate case: calibrate on the provided data without CV.
        cal = CalibratedClassifierCV(clf, method=cal_method, cv="prefit")
        cal.fit(X_fit, y_fit)
        return cal

    cv = min(5, min_class, n_samples)
    cv = max(cv, 2)
    cal = CalibratedClassifierCV(clf, method=cal_method, cv=cv)
    cal.fit(X_fit, y_fit)
    return cal


def evaluate_ppv_lcb_at_threshold(probs: np.ndarray, y: np.ndarray, tau: float) -> Dict[str, float]:
    if len(probs) != len(y):
        raise ValueError("probs and y must be same length")
    mask = np.isfinite(probs)
    probs = probs[mask]
    y = y[mask]
    if len(probs) == 0:
        return {"ppv": 0.0, "ppv_lcb": 0.0, "cov": 0.0}
    pred = probs >= float(tau)
    cov = float(pred.mean())
    pos = int(pred.sum())
    if pos == 0:
        return {"ppv": 0.0, "ppv_lcb": 0.0, "cov": cov}
    tp = int(((pred == 1) & (y == 1)).sum())
    ppv = tp / pos if pos else 0.0
    ppv_lcb = wilson_lcb(tp, pos)
    return {"ppv": float(ppv), "ppv_lcb": float(ppv_lcb), "cov": cov}
