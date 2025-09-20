import json, math
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score


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


def select_feature_cols(df: pd.DataFrame, inc, exc):
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in inc) and not any(c.startswith(p) for p in exc):
            cols.append(c)
    return cols


def build_feature_matrix(df: pd.DataFrame, cfg: dict, is_train: bool, bin_schema: dict | None):
    inc = list(cfg.get("features", {}).get("include_prefixes", ["qbin_","qbin_agebin_","since_","occ_"]))
    exc = list(cfg.get("features", {}).get("exclude_prefixes", []))
    nb  = int(cfg.get("n_bins_cont", 5))

    cols = select_feature_cols(df, inc, exc)
    cat_cols, cont_cols = [], []
    for c in cols:
        s = df[c]
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s) or c.startswith("qbin_"):
            cat_cols.append(c)
        else:
            cont_cols.append(c)

    if is_train:
        bin_schema = {"cont": {}}
        for c in cont_cols:
            edges = auto_bin_train(df.loc[:, c], nb)
            bin_schema["cont"][c] = edges.tolist()

    # assemble integer-coded categories (batch-add to avoid fragmentation)
    Xc = df[cat_cols].copy()
    newcols = {}
    for c in cont_cols:
        edges = np.array(bin_schema["cont"][c], dtype=float)
        newcols[f"q_{c}"] = apply_bins(df[c], edges)
    if newcols:
        Xc = pd.concat([Xc, pd.DataFrame(newcols, index=df.index)], axis=1, copy=False)

    for c in Xc.columns:
        Xc[c] = Xc[c].astype("Int64").astype("category")

    if is_train:
        # sklearn>=1.2 uses sparse_output; older uses sparse
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
        X = enc.fit_transform(Xc)
        return X, enc, bin_schema, Xc.columns.tolist()
    else:
        return Xc, None, bin_schema, Xc.columns.tolist()


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

    y = events_tr[["ts","outcome"]].copy()
    # exclude timeouts: keep only resolved outcomes
    y = y[y["outcome"].isin(["win","loss"])].copy()
    y["y"] = (y["outcome"] == "win").astype(int)
    df = cands_tr.merge(y[["ts","y"]], on="ts", how="inner")

    df_fit, df_val = split_train_valid_by_month(df, train_months)

    X_fit_skel, enc, bin_schema, cols = build_feature_matrix(df_fit, ml_cfg, is_train=True, bin_schema=None)
    # Build categorical-coded DataFrames for crosses (fit/val) with the same bin schema
    X_fit_cat, _, _, _ = build_feature_matrix(df_fit, ml_cfg, is_train=False, bin_schema=bin_schema)
    X_val_cat, _, _, _ = build_feature_matrix(df_val, ml_cfg, is_train=False, bin_schema=bin_schema)
    X_val = enc.transform(X_val_cat)
    y_fit = df_fit["y"].to_numpy()
    y_val = df_val["y"].to_numpy()

    used_crosses = []
    from scipy import sparse
    if bool(ml_cfg.get("use_motif_crosses", True)):
        gating_path = Path(out_dir) / "artifacts" / "gating.json"
        xfit_cross, used_crosses = build_motif_crosses(X_fit_cat, gating_path)
        xval_cross, _ = build_motif_crosses(X_val_cat, gating_path)
        if len(used_crosses):
            X_fit = sparse.hstack([X_fit_skel, xfit_cross.astype(np.float32).values], format="csr")
            X_val = sparse.hstack([X_val, xval_cross.astype(np.float32).values], format="csr")
        else:
            X_fit = X_fit_skel
    else:
        X_fit = X_fit_skel

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
            cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            cal.fit(X_val, y_val)
            p_val = cal.predict_proba(X_val)[:,1]
            ap = average_precision_score(y_val, p_val)
            best = max(best or (-1,None,None,{"C":C}), (ap, clf, cal, {"C":C}), key=lambda t:t[0])

    ap, clf, cal, params = best

    target_ppv = float(ml_cfg.get("target_ppv", 0.70)); z = 1.96
    p_val = cal.predict_proba(X_val.toarray() if model_name=="gbdt" else X_val)[:,1]
    dfv = pd.DataFrame({"p": p_val, "y": y_val}).sort_values("p", ascending=False)
    dfv["tp"] = dfv["y"].cumsum(); dfv["n"] = np.arange(1, len(dfv)+1)
    dfv["lcb"] = dfv.apply(lambda r: wilson_lcb(int(r.tp), int(r.n), z), axis=1)
    ok = dfv[dfv["lcb"] >= target_ppv]
    if len(ok):
        k = int(ok.index[0]) + 1; tau = float(dfv.iloc[k-1]["p"]); cov = float(k/len(dfv)); ach = float(dfv.iloc[k-1]["lcb"])
    else:
        min_cov = float(ml_cfg.get("min_coverage_frac", 0.02))
        k = max(1, int(min_cov * max(1,len(dfv)))); tau = float(dfv.iloc[k-1]["p"]); cov = float(k/len(dfv)); ach = float(dfv.iloc[k-1]["lcb"])

    art_dir = Path(out_dir) / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, art_dir / f"ml_{model_name}_base.pkl")
    joblib.dump(cal, art_dir / f"ml_{model_name}_calibrated.pkl")
    joblib.dump(enc, art_dir / "ml_encoder.pkl")
    with open(art_dir / "ml_bins.json","w",encoding="utf-8") as f: json.dump(bin_schema, f, ensure_ascii=False, indent=2)
    with open(art_dir / "ml_meta.json","w",encoding="utf-8") as f:
        json.dump({
            "model": model_name, "params": params, "ap_val": float(ap),
            "tau": tau, "target_ppv": target_ppv, "achieved_lcb": ach, "coverage_val": cov,
            "train_months": train_months, "used_crosses": used_crosses
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
    X  = enc.transform(Xc)

    used_crosses = meta.get("used_crosses", [])
    if used_crosses:
        gating_path = art / "gating.json"
        x_cross, _ = build_motif_crosses(Xc, gating_path)
        from scipy import sparse
        X = sparse.hstack([X, x_cross.astype(np.float32).values], format="csr")

    p = cal.predict_proba(X.toarray() if model=="gbdt" else X)[:,1]
    return pd.Series(p, index=cands_df.index, name="pwin")


def schedule_non_overlapping(df: pd.DataFrame, weight_mode: str = "expR", r_mult: float = 5.0):
    if "end_ts" in df.columns:
        end_ts = df["end_ts"].to_numpy()
    elif "outcome_ts" in df.columns:
        end_ts = df["outcome_ts"].to_numpy()
    else:
        return df

    ts = df["ts"].to_numpy()
    order = np.argsort(end_ts, kind="mergesort")
    ts, end_ts = ts[order], end_ts[order]
    if "pwin" in df.columns:
        p = df["pwin"].to_numpy()[order]
    else:
        p = np.ones_like(ts, dtype=float)
    w = (float(r_mult)*p - (1.0-p)) if weight_mode == "expR" else p

    import bisect
    pidx = [bisect.bisect_right(end_ts, ts[i]) - 1 for i in range(len(ts))]

    n = len(ts); dp = np.zeros(n+1, dtype=float); choose = np.zeros(n, dtype=bool)
    for i in range(1, n+1):
        skip = dp[i-1]
        take = w[i-1] + (dp[pidx[i-1]+1] if pidx[i-1] >= 0 else 0.0)
        if take > skip: dp[i]=take; choose[i-1]=True
        else: dp[i]=skip

    sel = []; i = n
    while i > 0:
        if choose[i-1]: sel.append(i-1); i = pidx[i-1]+1
        else: i -= 1
    sel = np.array(sorted(sel))
    return df.iloc[order[sel]].copy()
