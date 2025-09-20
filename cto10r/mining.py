import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .util import fit_quantile_edges


def wilson_lcb(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower confidence bound for a binomial proportion."""
    if n <= 0:
        return 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4 * n * n))) / denom
    return max(0.0, center - margin)

def wilson_ucb(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4 * n * n))) / denom
    return min(1.0, center + margin)


def canonical_rule_id(conds: list[dict]) -> str:
    """Build a stable rule identifier from a list of conditions."""
    if not conds:
        return ""
    key = lambda c: (c["feat"], c["op"], float(c["thr"]))
    parts = []
    for cond in sorted(conds, key=key):
        thr = "{:.6g}".format(float(cond["thr"]))
        parts.append(f"{cond['feat']}{cond['op']}{thr}")
    return " & ".join(parts)


def rule_precision_counts(sub: pd.DataFrame, include_timeouts: bool) -> tuple[int, int, int, int]:
    wins = int(sub["is_win"].sum())
    losses = int(sub["is_loss"].sum())
    timeouts = int(sub["is_to"].sum())
    if include_timeouts:
        n = int(len(sub))
    else:
        n = int(wins + losses)
    return n, wins, losses, timeouts



def prepare_literal_buckets(df: pd.DataFrame, train_mask, rules_cfg: dict, features_cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    auto = bool(rules_cfg.get("auto_binning", True))
    work = df.copy()
    if isinstance(train_mask, pd.Series):
        mask = train_mask.to_numpy()
    else:
        mask = np.asarray(train_mask, dtype=bool)
    if mask.shape[0] != len(work):
        raise ValueError("train_mask must match dataframe length")
    if not auto:
        return work, []
    age_cfg = (features_cfg.get("age", {}) or {})
    n_bins_feature = int(age_cfg.get("n_bins_feature", 5))
    n_bins_cont = int(rules_cfg.get("n_bins_cont", n_bins_feature))
    bin_cols = [c for c in work.columns if c.startswith("qbin_")]
    cont_cols = [c for c in work.columns if c.startswith(("since_", "occ_"))]

    # Batch-build B_* columns to avoid fragmentation
    newcols = {}
    for col in bin_cols:
        newcols[f"B_{col}"] = pd.to_numeric(work[col], errors="coerce")
    for col in cont_cols:
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy()
        bins = np.full(len(vals), np.nan, dtype=float)
        if len(vals):
            valid = np.isfinite(vals)
            train_valid = valid & mask
            edges = fit_quantile_edges(pd.Series(vals[train_valid]), n_bins_cont)
            if valid.any():
                bvals = np.clip(np.digitize(vals[valid], edges, right=False) - 1, 0, len(edges) - 2)
                bins[valid] = bvals.astype(float)
        newcols[f"B_{col}"] = pd.Series(bins)

    if newcols:
        work = pd.concat([work, pd.DataFrame(newcols, index=work.index)], axis=1, copy=False)
        bcols = list(newcols.keys())
    else:
        bcols = [c for c in work.columns if c.startswith("B_")]
    return work, bcols


def ensure_artifacts(out_dir: Path) -> Path:
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def mine_rules(cands: pd.DataFrame, events: pd.DataFrame, cfg: dict):
    '''Mine high-precision rules using Wilson LCB and per-month consistency.'''
    rules_cfg = (cfg.get("rules") or {})
    features_cfg = (cfg.get("features") or {})
    out_dir = Path(cfg.get("out_dir", "."))
    artifacts_dir = ensure_artifacts(out_dir)

    stats_columns = [
        "rule_id",
        "support_n",
        "wins",
        "losses",
        "timeouts",
        "precision",
        "precision_lcb",
        "precision_ucb",
        "resolve_frac",
        "timeout_rate",
        "lift",
        "lift_lcb",
        "months_with_lift",
        "unique_months",
    ]

    def emit_empty() -> dict:
        payload = {"promoted": []}
        (artifacts_dir / "gating.json").write_text(json.dumps(payload, indent=2))
        pd.DataFrame(columns=stats_columns).to_csv(artifacts_dir / "rules_eval.csv", index=False)
        return payload

    join_cols = ["ts", "side", "entry", "level", "risk_dist"]
    df = cands.merge(events, on=join_cols, how="inner", suffixes=("", ""))
    if "ym" not in df.columns:
        ts_series = pd.to_numeric(df["ts"], errors="coerce")
        df["ym"] = pd.to_datetime(ts_series, unit="ms", errors="coerce").dt.to_period("M").astype(str)

    if len(df) == 0:
        return emit_empty()

    df = df.copy()
    df["is_win"] = (df["outcome"] == "win").astype(int)
    df["is_loss"] = (df["outcome"] == "loss").astype(int)
    df["is_to"] = (df["outcome"] == "timeout").astype(int)

    include_timeouts = bool(rules_cfg.get("precision_include_timeouts", False))
    # Baseline metrics (TRAIN only)
    base_total = int(len(df))
    base_resolved = int((df["is_win"] + df["is_loss"]).sum())
    if include_timeouts:
        base_n = base_total
    else:
        base_n = base_resolved
    base_k = int(df["is_win"].sum())
    base_p = base_k / base_n if base_n > 0 else 0.0  # baseline_precision
    base_resolve = (base_resolved / base_total) if base_total > 0 else 0.0

    auto = bool(rules_cfg.get("auto_binning", True))
    max_terms = max(1, int(rules_cfg.get("max_terms", 2)))
    metric_name = str(rules_cfg.get("metric", "precision_lcb")).lower()
    min_frac = float(rules_cfg.get("min_support_frac", 0.0))
    n_train = len(df)
    min_support_n = max(int(math.ceil(min_frac * n_train)), 1) if n_train > 0 else 1
    min_months = max(1, int(rules_cfg.get("min_months", 1)))
    min_wins = int(rules_cfg.get("min_wins", 0))
    uplift_abs = float(rules_cfg.get("min_precision_uplift_abs", 0.0))
    uplift_mul = float(rules_cfg.get("min_lift_lcb", 0.0))
    # Strictness knobs (data-relative)
    min_precision_uplift_pp = float(rules_cfg.get("min_precision_uplift_pp", 0.0))
    min_resolve_uplift_pp = float(rules_cfg.get("min_resolve_uplift_pp", 0.0))
    max_timeout_rate = rules_cfg.get("max_timeout_rate", None)
    min_resolve_frac = rules_cfg.get("min_resolve_frac", None)
    abs_min_plcb = rules_cfg.get("min_precision_lcb", None)
    # loser mining knobs
    mine_losers = bool(rules_cfg.get("mine_losers", False))
    loser_top_k = int(rules_cfg.get("loser_top_k", 0))
    loser_max_precision_ucb = rules_cfg.get("loser_max_precision_ucb", None)
    loser_min_resolve_frac = rules_cfg.get("loser_min_resolve_frac", None)
    loser_min_months = int(rules_cfg.get("loser_min_months", 1))
    top_k = int(rules_cfg.get("top_k", 25))
    z_score = float(rules_cfg.get("wilson_z", 1.96))

    print(f"[mining] auto_binning={auto} n_train={n_train} min_support={min_support_n}", flush=True)
    print(f"[mining] baseline: precision={base_p:.3f} resolve={base_resolve:.3f}", flush=True)

    train_mask = np.ones(len(df), dtype=bool)
    work = df.copy()
    bin_edges: dict[str, np.ndarray] = {}
    feature_cols: list[str] = []

    if auto:
        work, bcols = prepare_literal_buckets(df, train_mask, rules_cfg, features_cfg)
        literal_cols = [c for c in sorted(bcols) if pd.to_numeric(work[c], errors="coerce").notna().any()]
        literal_pairs: list[tuple[str, str]] = []
    else:
        feature_cols = [
            "t_240",
            "t_60",
            "t_15",
            "accel_15_240",
            "body_dom",
            "atr_p",
            "dcw_p",
            "close_to_hi_atr",
            "close_to_lo_atr",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            return emit_empty()
        bins_cfg = rules_cfg.get("bins", {}) or {}
        for feat in feature_cols:
            if feat.startswith("t_"):
                edges_cfg = bins_cfg.get("t", [])
            elif feat == "accel_15_240":
                edges_cfg = bins_cfg.get("accel", [])
            elif feat == "body_dom":
                edges_cfg = bins_cfg.get("body_dom", [])
            elif feat in ("atr_p", "dcw_p"):
                edges_cfg = bins_cfg.get("atr_p", [])
            else:
                edges_cfg = bins_cfg.get("close_to_ext_atr", [])
            edges = [-np.inf] + list(edges_cfg) + [np.inf]
            bin_edges[feat] = np.array(edges, dtype=float)
            values = pd.to_numeric(work[feat], errors="coerce")
            bvals = np.full(len(values), np.nan)
            mask_vals = values.notna()
            if mask_vals.any():
                bvals[mask_vals] = np.digitize(values[mask_vals], bin_edges[feat], right=True)
            work[f"B_{feat}"] = bvals
        literal_cols = [f"B_{feat}" for feat in feature_cols if pd.notna(work[f"B_{feat}"]).any()]
        literal_pairs = [(feat, f"B_{feat}") for feat in feature_cols if f"B_{feat}" in literal_cols]

    if not literal_cols:
        return emit_empty()

    progress_on = bool(cfg.get("progress", True))

    rule_stats: dict[str, dict] = {}
    rule_payload: dict[str, dict] = {}

    z_score = float(rules_cfg.get("wilson_z", 1.96))

    def register_rule(rule_id: str, conds: list[dict], sub: pd.DataFrame) -> None:
        n, wins, losses, tos = rule_precision_counts(sub, include_timeouts)
        if n < min_support_n:
            return
        if wins < min_wins:
            return
        unique_months = int(sub["ym"].nunique()) if "ym" in sub.columns else 1
        if unique_months < min_months:
            return
        p_hat = wins / n if n else 0.0
        p_lcb = wilson_lcb(wins, n, z=z_score)
        if base_p > 0:
            lift = p_hat / base_p
            lift_lcb = p_lcb / base_p
        else:
            lift = float("inf") if p_hat > 0 else 0.0
            lift_lcb = float("inf") if p_lcb > 0 else 0.0

        months_with_lift = 0
        if "ym" in sub.columns:
            bym = sub.groupby("ym").agg(
                n=("is_win", "size"),
                win=("is_win", "sum"),
                loss=("is_loss", "sum"),
                to=("is_to", "sum"),
            ).reset_index()
            if include_timeouts:
                denom = bym["n"].astype(float).replace(0.0, np.nan)
            else:
                denom = (bym["win"] + bym["loss"]).astype(float).replace(0.0, np.nan)
            probs = bym["win"] / denom
            if base_p > 0:
                months_with_lift = int((probs > base_p).sum())
            else:
                months_with_lift = int((bym["win"] > 0).sum())

        # Resolve/timeout fractions always computed on total sub size
        total_sub = int(len(sub))
        resolved_n = int(wins + losses)
        resolve_frac = (resolved_n / total_sub) if total_sub > 0 else 0.0
        timeout_rate = (int(tos) / total_sub) if total_sub > 0 else 0.0
        # Precision UCB for loser mining: use same n as precision denominator
        p_ucb = wilson_ucb(wins, n, z=z_score)

        metrics = {
            "rule_id": rule_id,
            "support_n": int(n),
            "wins": int(wins),
            "losses": int(losses),
            "timeouts": int(tos),
            "precision": float(p_hat),
            "precision_lcb": float(p_lcb),
            "precision_ucb": float(p_ucb),
            "lift": float(lift),
            "lift_lcb": float(lift_lcb),
            "resolve_frac": float(resolve_frac),
            "timeout_rate": float(timeout_rate),
            "months_with_lift": int(months_with_lift),
            "unique_months": int(unique_months),
        }
        existing = rule_stats.get(rule_id)
        replace = False
        if existing is None:
            replace = True
        else:
            if metrics["precision_lcb"] > existing["precision_lcb"]:
                replace = True
            elif metrics["precision_lcb"] == existing["precision_lcb"] and metrics["support_n"] > existing["support_n"]:
                replace = True
        if replace:
            rule_stats[rule_id] = metrics
            rule_payload[rule_id] = {
                "rule_id": rule_id,
                "conds": conds,
                "precision": metrics["precision"],
                "precision_lcb": metrics["precision_lcb"],
                "support_n": metrics["support_n"],
                "wins": metrics["wins"],
                "losses": metrics["losses"],
                "timeouts": metrics["timeouts"],
                "lift": metrics["lift"],
                "lift_lcb": metrics["lift_lcb"],
                "unique_months": metrics["unique_months"],
                "months_with_lift": metrics["months_with_lift"],
            }

    if auto:
        iter_1d = tqdm(literal_cols, desc="mine 1D", dynamic_ncols=True, leave=False) if progress_on else literal_cols
        for col in iter_1d:
            mask_col = work[col].notna()
            if not mask_col.any():
                continue
            for val, sub in work[mask_col].groupby(col, observed=True):
                conds = [{"feat": col, "op": "==", "thr": float(val)}]
                register_rule(canonical_rule_id(conds), conds, sub)
        if progress_on and hasattr(iter_1d, "close"):
            iter_1d.close()
        if max_terms >= 2:
            pairs = []
            for i in range(len(literal_cols)):
                for j in range(i + 1, len(literal_cols)):
                    col_i = literal_cols[i]
                    col_j = literal_cols[j]
                    if col_i == col_j:
                        continue
                    pairs.append((col_i, col_j))
            iter_2d = tqdm(pairs, desc="mine 2D", dynamic_ncols=True, leave=False) if progress_on else pairs
            for col_i, col_j in iter_2d:
                mask_pair = work[col_i].notna() & work[col_j].notna()
                if not mask_pair.any():
                    continue
                for (val_i, val_j), sub in work[mask_pair].groupby([col_i, col_j], observed=True):
                    conds = [
                        {"feat": col_i, "op": "==", "thr": float(val_i)},
                        {"feat": col_j, "op": "==", "thr": float(val_j)},
                    ]
                    register_rule(canonical_rule_id(conds), conds, sub)
            if progress_on and hasattr(iter_2d, "close"):
                iter_2d.close()
    else:
        iter_1d_pairs = tqdm(literal_pairs, desc="mine 1D", dynamic_ncols=True, leave=False) if progress_on else literal_pairs
        for feat, col in iter_1d_pairs:
            mask_col = work[col].notna()
            if not mask_col.any():
                continue
            for val, sub in work[mask_col].groupby(col, observed=True):
                conds = bin_to_conditions(feat, val, bin_edges[feat])
                if not conds:
                    continue
                register_rule(canonical_rule_id(conds), conds, sub)
        if progress_on and hasattr(iter_1d_pairs, "close"):
            iter_1d_pairs.close()
        if max_terms >= 2:
            pairs = []
            for i in range(len(literal_pairs)):
                feat_i, col_i = literal_pairs[i]
                for j in range(i + 1, len(literal_pairs)):
                    feat_j, col_j = literal_pairs[j]
                    if col_i == col_j:
                        continue
                    pairs.append((feat_i, col_i, feat_j, col_j))
            iter_2d_pairs = tqdm(pairs, desc="mine 2D", dynamic_ncols=True, leave=False) if progress_on else pairs
            for feat_i, col_i, feat_j, col_j in iter_2d_pairs:
                mask_pair = work[col_i].notna() & work[col_j].notna()
                if not mask_pair.any():
                    continue
                for (val_i, val_j), sub in work[mask_pair].groupby([col_i, col_j], observed=True):
                    conds = bin_to_conditions(feat_i, val_i, bin_edges[feat_i]) + bin_to_conditions(feat_j, val_j, bin_edges[feat_j])
                    if not conds:
                        continue
                    register_rule(canonical_rule_id(conds), conds, sub)
            if progress_on and hasattr(iter_2d_pairs, "close"):
                iter_2d_pairs.close()

    if not rule_stats:
        return emit_empty()

    stats = pd.DataFrame(rule_stats.values()).reindex(columns=stats_columns)

    metric_map = {
        "precision": "precision",
        "precision_lcb": "precision_lcb",
        "lift": "lift",
        "lift_lcb": "lift_lcb",
        "support": "support_n",
        "support_n": "support_n",
    }
    metric_col = metric_map.get(metric_name, "precision_lcb")

    # Thresholds
    stats = stats.sort_values([metric_col, "support_n"], ascending=[False, False]).reset_index(drop=True)

    prec_threshold = (base_p + uplift_abs) if base_p > 0 else uplift_abs
    lift_threshold = max(uplift_mul, 0.0)
    # Data-relative uplifts in percentage points
    prec_pp = base_p + (min_precision_uplift_pp / 100.0)
    res_pp = base_resolve + (min_resolve_uplift_pp / 100.0)

    promoted = stats.copy()
    promoted = promoted[(promoted["precision_lcb"] >= max(prec_threshold, prec_pp)) & (promoted["lift_lcb"] >= lift_threshold)]
    # Resolve requirements: both absolute floor and uplift over baseline
    promoted = promoted[promoted["resolve_frac"] >= res_pp]
    if min_resolve_frac is not None:
        promoted = promoted[promoted["resolve_frac"] >= float(min_resolve_frac)]
    if max_timeout_rate is not None:
        promoted = promoted[promoted["timeout_rate"] <= float(max_timeout_rate)]
    if abs_min_plcb is not None:
        promoted = promoted[promoted["precision_lcb"] >= float(abs_min_plcb)]
    if min_wins > 0:
        promoted = promoted[promoted["wins"] >= min_wins]
    promoted = promoted.head(top_k).reset_index(drop=True)

    promoted_ids = promoted["rule_id"].tolist()
    promoted_full = [rule_payload[rid] for rid in promoted_ids if rid in rule_payload]

    # Loser blocklist mining
    losers_payload = []
    if mine_losers and loser_top_k > 0 and len(stats):
        losers = stats.copy()
        if loser_min_resolve_frac is not None:
            losers = losers[losers["resolve_frac"] >= float(loser_min_resolve_frac)]
        losers = losers[losers["unique_months"] >= int(loser_min_months)]
        if loser_max_precision_ucb is not None:
            losers = losers[losers["precision_ucb"] <= float(loser_max_precision_ucb)]
        losers = losers.sort_values(["precision_ucb", "support_n"], ascending=[True, False]).head(loser_top_k)
        loser_ids = losers["rule_id"].tolist()
        losers_payload = [rule_payload[rid] for rid in loser_ids if rid in rule_payload]

    payload = {"promoted": promoted_full, "blocklist": losers_payload}
    (artifacts_dir / "gating.json").write_text(json.dumps(payload, indent=2))
    stats.to_csv(artifacts_dir / "rules_eval.csv", index=False)
    print(f"[mining] promoted={len(promoted_full)} losers={len(losers_payload)}", flush=True)

    return payload
