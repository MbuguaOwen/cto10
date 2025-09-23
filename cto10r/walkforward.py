# cto10r/walkforward.py


from __future__ import annotations


import argparse


import json


import shutil
import copy


from dataclasses import dataclass


from pathlib import Path


from typing import Any, Dict, Iterable, List, Optional, Tuple


import numpy as np


import pandas as pd


import yaml
from tqdm.auto import tqdm


# --- project imports that exist in your repo ---


from .io_utils import ensure_dir, write_json, skip_if_exists


from .util import exception_to_report, infer_bar_seconds, safe_div, dump_json, wilson_lcb


from .bars import load_bars_any, build_features, fit_percentiles, apply_percentiles, add_age_features, add_nonlinear_features


from .candidates import build_candidates_router


from .ticks import iter_ticks_files, stream_ticks_window, label_events_from_ticks


from .mining import mine_rules, prepare_literal_buckets
from .ml import schedule_non_overlapping
from .gate import (
    GateConfig,
    LoserMaskConfig,
    train_gate,
    load_trained_gate,
    literalize_candidates,
)


# ------------------------ constants ------------------------


RULE_FINGERPRINT_FEATURES = [


    "t_240", "t_60", "t_15",


    "accel_15_240",


    "body_dom",


    "close_to_ext_atr",


    "atr_p", "dcw_p",


    "eta",


    "ym",


]


ALL_STAGES = ["features", "tick_labeling", "mining", "simulate"]


# ------------------------ small utils ------------------------


def _log(msg: str) -> None:


    print(msg, flush=True)




def _pbar(seq, desc: str, enabled: bool, total: int | None = None):


    if not enabled:


        return seq


    return tqdm(seq, desc=desc, total=total, dynamic_ncols=True, leave=False)



def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:


    out = df.copy()


    if "ts" in out.columns:


        out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")


    if "side" in out.columns:


        out["side"] = out["side"].astype(str)


    for c in ["entry", "level", "risk_dist"]:


        if c in out.columns:


            out[c] = pd.to_numeric(out[c], errors="coerce")


    return out


def month_list(start: str, end: str) -> List[str]:


    idx = pd.period_range(start=start, end=end, freq="M")


    return [str(p) for p in idx.astype(str)]


def make_folds(months: List[str], train: int, test: int, step: int) -> List[Tuple[List[str], List[str]]]:


    folds: List[Tuple[List[str], List[str]]] = []


    i = 0


    while True:


        tr = months[i : i + train]


        te = months[i + train : i + train + test]


        if len(te) < test:


            break


        folds.append((tr, te))


        i += step


    return folds


def rule_hit_row(row: pd.Series, conds: List[Dict[str, Any]]) -> bool:


    for c in conds:


        feat = c["feat"]; op = c["op"]; thr = float(c["thr"])


        if feat not in row or pd.isna(row[feat]): 


            return False


        x = float(row[feat])

        if op == "==" and not (x == thr): return False


        if op == ">=" and not (x >= thr): return False


        if op == "<=" and not (x <= thr): return False


        if op == ">"  and not (x >  thr): return False


        if op == "<"  and not (x <  thr): return False


    return True


def assign_best_rule_id(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> pd.Series:


    out = pd.Series([None]*len(df), index=df.index, dtype="object")


    for r in rules:


        rid = r.get("rule_id"); conds = r.get("conds", [])


        if rid is None or not conds: 


            continue


        mask = df.apply(lambda row: rule_hit_row(row, conds), axis=1)


        out.loc[mask & out.isna()] = rid


        if out.notna().all():


            break


    return out


# ------------------------ CLI ------------------------


@dataclass


class Args:


    config: str


    mode: str


    force: bool


    clean: bool


    only: Optional[str]


    from_stage: Optional[str]


    until_stage: Optional[str]


def parse_args() -> Args:


    p = argparse.ArgumentParser()


    p.add_argument("--config", required=True)


    p.add_argument("--mode", required=True, choices=["preflight", "walkforward"])


    p.add_argument("--force", action="store_true")


    p.add_argument("--clean", action="store_true")


    p.add_argument("--only", choices=ALL_STAGES)


    p.add_argument("--from", dest="from_stage", choices=ALL_STAGES)


    p.add_argument("--until", dest="until_stage", choices=ALL_STAGES)


    a = p.parse_args()


    return Args(


        config=a.config, mode=a.mode, force=a.force, clean=a.clean,


        only=a.only, from_stage=a.from_stage, until_stage=a.until_stage


    )


def load_cfg(path: str) -> Dict[str, Any]:


    with open(path, "r") as f:


        return yaml.safe_load(f)


def stages_to_run(only: Optional[str], from_stage: Optional[str], until_stage: Optional[str]) -> List[str]:


    if only:


        return [only]


    if from_stage or until_stage:


        start = ALL_STAGES.index(from_stage) if from_stage else 0


        end = ALL_STAGES.index(until_stage) if until_stage else len(ALL_STAGES) - 1


        return ALL_STAGES[start : end + 1]


    return ALL_STAGES[:]


# ------------------------ bars loader (no missing imports) ------------------------


def _build_bar_paths(sym: str, months: List[str], data_cfg: Dict[str, Any]) -> List[Path]:


    """Create a list of monthly bar file paths using YAML file_patterns."""


    bars_dir = Path(data_cfg["bars_dir"])


    pat = data_cfg["file_patterns"]["bars"]  # e.g. "{SYMBOL}/{SYMBOL}-1m-{YYYY}-{MM}.csv"


    paths: List[Path] = []


    for ym in months:


        y, m = ym.split("-")


        rel = pat.replace("{SYMBOL}", sym).replace("{YYYY}", y).replace("{MM}", m)


        paths.append(bars_dir / rel)


    return paths


def _load_bars_months(sym: str, months: List[str], data_cfg: Dict[str, Any], bar_schema: str, progress_on: bool) -> pd.DataFrame:


    """Read monthly bar files and concatenate; add 'ym' derived from ts."""


    paths = _build_bar_paths(sym, months, data_cfg)


    dfs: List[pd.DataFrame] = []


    for p in _pbar(paths, desc=f"{sym} bars", enabled=progress_on, total=len(paths)):


        if not p.exists():


            continue


        df = load_bars_any(p, schema=bar_schema)


        dfs.append(df)


    if not dfs:


        return pd.DataFrame(columns=["ts","open","high","low","close","ym"])


    bars = pd.concat(dfs, ignore_index=True).sort_values("ts")


    # derive ym (YYYY-MM) from timestamp (ms)


    ts = pd.to_datetime(bars["ts"], unit="ms", utc=True)


    bars["ym"] = ts.dt.strftime("%Y-%m")


    return bars


# ------------------------ stages ------------------------


def stage_features(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], fold_dir: Path, force: bool) -> None:


    _log(f"[{sym}] features: start train={train_months} test={test_months}")


    ensure_dir(fold_dir)


    cfg_echo = fold_dir / "cfg_echo.json"


    if not cfg_echo.exists():


        write_json(cfg_echo, cfg)


    cpath = fold_dir / "candidates.parquet"


    if skip_if_exists(cpath, force):


        _log(f"[{sym}] SKIP features (exists)")


        return


    data_cfg = cfg["data"]


    bar_schema = data_cfg.get("bar_schema", "auto")


    progress_on = bool(data_cfg.get("progress", cfg.get("io", {}).get("progress", True)))


    months_all = list(dict.fromkeys(train_months + test_months))


    bars = _load_bars_months(sym, months_all, data_cfg, bar_schema, progress_on)


    if bars.empty:


        pd.DataFrame().to_parquet(cpath, index=False)


        _log(f"[{sym}] features: candidates=0 (no bars)")


        return


    bars.sort_values("ts", inplace=True)


    f_cfg = cfg["features"]


    feat_all = build_features(


        bars,


        f_cfg["atr_n"],


        f_cfg["donch_n_candidate"],


        f_cfg.get("horizons_min", []),


    )


    fit_on = f_cfg.get("percentiles_fit_on", "train")


    if fit_on == "train":


        if "ym" in feat_all.columns:


            train_slice = feat_all[feat_all["ym"].isin(train_months)]


        else:


            train_slice = feat_all


        if len(train_slice):


            f_atr, f_dcw = fit_percentiles(train_slice)


            feat_all = apply_percentiles(feat_all, f_atr, f_dcw)


    elif fit_on == "all":


        if len(feat_all):


            f_atr, f_dcw = fit_percentiles(feat_all)


            feat_all = apply_percentiles(feat_all, f_atr, f_dcw)

    # Add non-linear features after base + percentiles
    feat_all = add_nonlinear_features(feat_all, cfg)


    feat_all = add_age_features(feat_all, cfg, train_months, fold_dir)

    cands = build_candidates_router(feat_all, cfg)


    cands = cands.sort_values("ts").reset_index(drop=True)
    # Drop duplicate columns to keep PyArrow happy
    cands = cands.loc[:, ~cands.columns.duplicated()]

    cands.to_parquet(cpath, index=False)


    _log(f"[{sym}] features: candidates={len(cands)}")


def stage_tick_labeling(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], fold_dir: Path, force: bool) -> None:


    _log(f"[{sym}] tick_labeling: start")


    epath = fold_dir / "events.parquet"


    if skip_if_exists(epath, force):


        _log(f"[{sym}] SKIP tick_labeling (exists)")


        return


    cands = pd.read_parquet(fold_dir / "candidates.parquet")


    if cands.empty:


        pd.DataFrame(columns=["ts", "side", "entry", "level", "risk_dist", "outcome"]).to_parquet(epath, index=False)


        _log(f"[{sym}] events: 0 (no candidates)")


        return


    cands = cands.sort_values("ts").reset_index(drop=True)


    _log(f"[{sym}] tick_labeling: candidates={len(cands)}")


    months_all = list(dict.fromkeys(train_months + test_months))


    try:


        bar_seconds = infer_bar_seconds(cands["ts"])


    except Exception:


        bar_seconds = 60


    labels_cfg = cfg["labels"]


    H_hours = int(labels_cfg.get("horizon_hours", 72))


    Q_hours = int(labels_cfg.get("quick_ignition_hours", 2))


    non_overlap = bool(labels_cfg.get("non_overlap", True))


    stop_band_atr = float(labels_cfg.get("stop_band_atr", 0.0))


    consec = int(labels_cfg.get("consec_ticks", labels_cfg.get("violation_consecutive_ticks", 1)))


    dwell_ms = int(labels_cfg.get("dwell_ms", labels_cfg.get("violation_dwell_ms", 0)))


    r_mult = float(labels_cfg.get("r_mult", 10.0))


    ts_cfg = labels_cfg.get("time_scale_from_eta")


    base_eta = None


    clamp = None


    if ts_cfg:


        base_eta = float(ts_cfg.get("base_eta", labels_cfg.get("eta_atr", 0.30)))


        clamp_vals = ts_cfg.get("clamp", [0.5, 20.0])


        clamp = (float(clamp_vals[0]), float(clamp_vals[1]))


    H_ms_base = int(H_hours * 3600 * 1000)


    if ts_cfg and clamp and "eta" in cands.columns:


        base = float(base_eta or labels_cfg.get("eta_atr", 0.30)) or 1.0


        scale = (cands["eta"].astype(float) / base).clip(clamp[0], clamp[1])


        H_ms_i = (H_ms_base * scale).astype("int64")


        tmin = int(cands["ts"].min())


        tmax = int((cands["ts"] + H_ms_i).max())


    else:


        tmin = int(cands["ts"].min())


        tmax = int(cands["ts"].max() + H_ms_base)


    ticks_dir = Path(cfg["data"]["ticks_dir"])


    patt = cfg["data"]["file_patterns"]["ticks"]


    tick_files = list(iter_ticks_files(ticks_dir, sym, months_all, patt))


    progress_on = bool(cfg.get("io", {}).get("progress", True))


    if not tick_files:


        ev = cands.copy()


        ev["outcome"] = "timeout"


        ev["outcome_ts"] = ev["ts"]


        ev["r1_ts"] = np.nan


        ev["tp_ts"] = np.nan


        ev.to_parquet(epath, index=False)


        _log(f"[{sym}] events: {len(ev)} (wins=0, losses=0, timeouts={len(ev)})")


        return


    def _ticks_gen(files, tmin_ms, tmax_ms):


        seq = _pbar(files, desc=f"{sym} ticks", enabled=progress_on, total=len(files))


        for _, path in seq:


            for chunk in stream_ticks_window(path, tmin_ms, tmax_ms):


                yield chunk


    ev = label_events_from_ticks(


        cands=cands,


        ticks_iterables=_ticks_gen(tick_files, tmin, tmax),


        bar_seconds=bar_seconds,


        horizon_hours=H_hours,


        quick_hours=Q_hours,


        non_overlap=non_overlap,


        stop_band_atr=stop_band_atr,


        consec_ticks=consec,


        dwell_ms=dwell_ms,


        base_eta_for_time_scale=base_eta,


        time_scale_clamp=clamp,


        r_mult=r_mult,


        show_progress=progress_on,


    )


    ev = ev.sort_values("ts").reset_index(drop=True)


    ev.to_parquet(epath, index=False)


    wins = int((ev["outcome"] == "win").sum())


    losses = int((ev["outcome"] == "loss").sum())


    timeouts = int((ev["outcome"] == "timeout").sum())


    _log(f"[{sym}] events: {len(ev)} (wins={wins}, losses={losses}, timeouts={timeouts})")


def stage_mining(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], fold_dir: Path, force: bool) -> None:


    _log(f"[{sym}] mining: start train={train_months}")


    apath = fold_dir / "artifacts" / "gating.json"


    if skip_if_exists(apath, force):


        _log(f"[{sym}] SKIP mining (exists)")


        return


    ensure_dir(fold_dir / "artifacts")


    cands = pd.read_parquet(fold_dir / "candidates.parquet")


    events = pd.read_parquet(fold_dir / "events.parquet")


    join_cols = ["ts", "side", "entry", "level", "risk_dist"]


    if "ym" in cands.columns:


        cands_tr = cands[cands["ym"].isin(train_months)].copy()


    else:


        cands_tr = cands.copy()


    if cands_tr.empty:


        events_tr = events.iloc[0:0].copy()


    else:


        events_tr = events.merge(cands_tr[join_cols], on=join_cols, how="inner")


    cands_tr = _norm_keys(cands_tr)


    events_tr = _norm_keys(events_tr)


    rules_cfg = cfg.get("mining", {}).get("rules", {})
    progress_on = bool(cfg.get("io", {}).get("progress", True))


    mining_payload, mined_df = mine_rules(
        cands_tr,
        events_tr,
        {"rules": rules_cfg, "out_dir": str(fold_dir), "features": cfg.get("features", {}), "progress": progress_on},
    )

    promoted = mining_payload.get("promoted", [])

    _log(f"[{sym}] mining: rows={len(events_tr)} promoted={len(promoted)}")

    gate_cfg_raw = cfg.get("gate", {}) or {}
    loser_cfg_raw = cfg.get("loser_mask", {}) or {}
    gate_cfg = GateConfig(
        calibration=str(gate_cfg_raw.get("calibration", "isotonic")),
        target_ppv_lcb=float(gate_cfg_raw.get("target_ppv_lcb", 0.60)),
        min_coverage=float(gate_cfg_raw.get("min_coverage", 0.03)),
        crosses_cap=int(gate_cfg_raw.get("crosses_cap", 30)),
        val_months=int(gate_cfg_raw.get("val_months", 1)),
    )
    loser_cfg = LoserMaskConfig(
        enabled=bool(loser_cfg_raw.get("enabled", True)),
        min_support=int(loser_cfg_raw.get("min_support", 40)),
        min_months_with_lift=int(loser_cfg_raw.get("min_months_with_lift", 2)),
        min_ppv_lcb=float(loser_cfg_raw.get("min_ppv_lcb", 0.55)),
        save_artifact=bool(loser_cfg_raw.get("save_artifact", True)),
    )

    gate_features_cfg = copy.deepcopy(cfg.get("ml_gating", {}))
    gate_features_cfg["gating_path"] = str(fold_dir / "artifacts" / "gating.json")

    labels = events_tr[events_tr["outcome"].isin(["win", "loss"])].copy()
    labels["y"] = (labels["outcome"] == "win").astype(int)
    if "ym" not in labels.columns and "ym" in events_tr.columns:
        labels["ym"] = events_tr.loc[labels.index, "ym"].to_numpy()
    if "ym" not in labels.columns and "ym" in cands_tr.columns:
        labels = labels.merge(
            cands_tr[["ts", "ym"]].drop_duplicates("ts"),
            on="ts",
            how="left",
            suffixes=("", "_cand"),
        )
        if "ym_cand" in labels.columns:
            labels["ym"] = labels["ym"].fillna(labels.pop("ym_cand"))

    label_cols = ["ts", "y"]
    if "ym" in labels.columns:
        label_cols.append("ym")
    all_train = cands_tr.merge(labels[label_cols], on="ts", how="inner")
    if "ym_x" in all_train.columns:
        all_train["ym"] = all_train.pop("ym_x")
    if "ym_y" in all_train.columns:
        all_train["ym"] = all_train.pop("ym_y")
    if "ym" not in all_train.columns:
        months_sorted = sorted(set(train_months))
        default_month = months_sorted[-1] if months_sorted else "unknown"
        all_train["ym"] = default_month
    all_train["ym"] = all_train["ym"].astype(str)

    val_months = []
    if gate_cfg.val_months > 0:
        months_sorted = sorted({str(m) for m in train_months})
        val_months = months_sorted[-gate_cfg.val_months :]
    val_mask = all_train["ym"].isin(val_months) if val_months else pd.Series(False, index=all_train.index)
    val_df = all_train[val_mask].copy()
    train_core = all_train[~val_mask].copy()
    if train_core.empty:
        train_core = all_train.copy()
    if val_df.empty:
        val_df = all_train.copy()

    artifacts_dir = fold_dir / "artifacts"

    if len(train_core) and train_core["y"].nunique() > 1:
        try:
            gate, diag = train_gate(
                train_core,
                val_df,
                gate_features_cfg,
                gate_cfg,
                loser_cfg,
                mined_df,
                artifacts_dir,
            )
            _log(
                f"[{sym}] gate: τ={diag['tau']:.4f} PPV={diag['ppv']:.3f} LCB={diag['ppv_lcb']:.3f} cov={diag['cov']:.3f} loser_rules={diag['loser_rules']}"
            )
        except Exception as e:
            _log(f"[{sym}] gate training failed: {e}")
    else:
        _log(f"[{sym}] gate: insufficient labeled data for training (rows={len(train_core)})")


def stage_simulate(
    sym: str,
    train_months: List[str],
    test_months: List[str],
    cfg: Dict[str, Any],
    fold_dir: Path,
    force: bool,
    fold_idx: Optional[int] = None,
) -> None:


    _log(f"[{sym}] simulate: start test={test_months}")


    spath = fold_dir / "stats.json"


    tpath = fold_dir / "trades.csv"


    trpath = fold_dir / "train_table.csv"


    if skip_if_exists(spath, force) and tpath.exists() and trpath.exists():


        _log(f"[{sym}] SKIP simulate (exists)")


        return


    cands = pd.read_parquet(fold_dir / "candidates.parquet")


    events = pd.read_parquet(fold_dir / "events.parquet")


    rules_cfg = (cfg.get("mining", {}).get("rules") or {})
    features_cfg = (cfg.get("features") or {})
    if "ym" in cands.columns:
        train_mask_all = cands["ym"].isin(train_months)
    else:
        train_mask_all = pd.Series(True, index=cands.index)
    cands, _ = prepare_literal_buckets(cands, train_mask_all, rules_cfg, features_cfg)


    labels_cfg_sim = cfg.get("labels", {}) or {}
    label_r_mult = float(labels_cfg_sim.get("r_mult", 5.0))


    join_cols = ["ts", "side", "entry", "level", "risk_dist"]


    if "ym" in cands.columns:


        cands_t = cands[cands["ym"].isin(test_months)].copy()


    else:


        cands_t = cands.copy()


    if cands_t.empty:


        cands_t_norm = _norm_keys(cands_t)


        ev = events.iloc[0:0].copy()


    else:


        cands_t_norm = _norm_keys(cands_t)


        ev = events.merge(cands_t[join_cols], on=join_cols, how="inner")


    ev = _norm_keys(ev.copy())


    feature_cols = [c for c in RULE_FINGERPRINT_FEATURES if c in cands_t_norm.columns]


    literal_cols = [c for c in cands_t_norm.columns if c.startswith("B_")]
    age_cols = [c for c in cands_t_norm.columns if c.startswith("qbin_agebin_")]
    merge_cols = list(dict.fromkeys(join_cols + feature_cols + literal_cols + age_cols))


    if not ev.empty:


        ev = ev.merge(cands_t_norm[merge_cols], on=join_cols, how="left")


    rules = []
    gpath = fold_dir / "artifacts" / "gating.json"
    if gpath.exists():
        with open(gpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            rules = data.get("promoted", [])
    ev["rule_id"] = assign_best_rule_id(ev, rules) if rules else None

    gate_bundle = fold_dir / "artifacts" / "gate_bundle.pkl"
    gate_features_cfg = copy.deepcopy(cfg.get("ml_gating", {}))
    gate_features_cfg["gating_path"] = str(gpath)
    gate_cfg_section = (cfg.get("gate", {}) or {})
    tau_floor = float(gate_cfg_section.get("tau_floor", 0.0))
    tau_override = gate_cfg_section.get("tau_override", None)
    gate_tau = float("nan")
    coverage = 0.0
    empirical_ppv = float("nan")
    gate = None
    if gate_bundle.exists():
        try:
            gate = load_trained_gate(gate_bundle)
            tau_raw = getattr(gate, "tau", None)
            tau = tau_raw
            if tau_override not in (None, ""):
                try:
                    tau = float(tau_override)
                except Exception:
                    tau = tau_raw
            elif tau is not None and tau_floor > 0.0:
                try:
                    tau_val = float(tau)
                except Exception:
                    tau_val = None
                if tau_val is not None:
                    tau = tau_floor if np.isnan(tau_val) else max(tau_val, tau_floor)
            if tau is not None:
                gate.tau = float(tau)
            gate_tau = gate.tau
            _log(f"[{sym}] gate: tau_used={gate_tau:.4f}")
            X_test, y_test, literals = literalize_candidates(cands_t_norm, gate_features_cfg, gate.feature_meta)
            p_vals = gate.score(X_test)
            losers_fire = gate.loser_mask_vec(literals)
            enter_mask = (p_vals >= gate.tau) & (~losers_fire)
            cands_t_norm = cands_t_norm.copy()
            cands_t_norm["p_win"] = p_vals
            cands_t_norm["loser_mask"] = losers_fire
            cands_t_norm["enter"] = enter_mask
            cands_t_norm["tau_used"] = gate.tau
            gate_tau = gate.tau
        except Exception as e:
            _log(f"[{sym}] gate scoring failed: {e}")
    if "p_win" not in cands_t_norm.columns:
        cands_t_norm = cands_t_norm.copy()
        cands_t_norm["p_win"] = np.nan
        cands_t_norm["loser_mask"] = False
        cands_t_norm["enter"] = False
        cands_t_norm["tau_used"] = gate_tau

    ev = ev.merge(
        cands_t_norm[["ts", "p_win", "loser_mask", "enter", "tau_used"]],
        on="ts",
        how="left",
    )
    ev["p_win"] = pd.to_numeric(ev["p_win"], errors="coerce")
    ev["loser_mask"] = ev["loser_mask"].fillna(False)
    ev["enter"] = ev["enter"].fillna(False)
    ev["tau_used"] = ev["tau_used"].fillna(gate_tau)

    ev = ev.sort_values("ts").reset_index(drop=True)

    if "ts" in ev.columns:
        ev["ts"] = pd.to_numeric(ev["ts"], errors="coerce")
    if "end_ts" not in ev.columns and "outcome_ts" in ev.columns:
        ev["end_ts"] = pd.to_numeric(ev["outcome_ts"], errors="coerce")
    elif "end_ts" in ev.columns:
        ev["end_ts"] = pd.to_numeric(ev["end_ts"], errors="coerce")
    else:
        ev["end_ts"] = pd.Series(np.nan, index=ev.index, dtype="float64")
    if "pwin" not in ev.columns and "p_win" in ev.columns:
        ev["pwin"] = pd.to_numeric(ev["p_win"], errors="coerce")

    decision_df = cands_t_norm.copy()
    default_enter = pd.Series(False, index=decision_df.index, dtype=bool)
    go_series = (
        pd.to_numeric(decision_df.get("enter", default_enter), errors="coerce")
        .fillna(0)
        .astype(bool)
    )

    label_merge_cols = [c for c in join_cols if c in decision_df.columns and c in ev.columns]
    if label_merge_cols:
        label_lookup = ev[label_merge_cols + ["outcome"]].copy()
        label_lookup["label_resolved"] = label_lookup["outcome"].map({
            "win": 1,
            "loss": -1,
            "timeout": 0,
        })
        label_lookup = label_lookup.drop_duplicates(subset=label_merge_cols, keep="last")
        decision_df = decision_df.merge(
            label_lookup[label_merge_cols + ["label_resolved"]],
            on=label_merge_cols,
            how="left",
        )
        if "label" in decision_df.columns:
            decision_df["label"] = pd.to_numeric(decision_df["label"], errors="coerce")
            decision_df["label"] = decision_df["label"].fillna(decision_df.pop("label_resolved"))
        else:
            decision_df["label"] = decision_df.pop("label_resolved")
    else:
        decision_df["label"] = pd.to_numeric(
            decision_df.get("label", pd.Series(np.nan, index=decision_df.index)),
            errors="coerce",
        )

    decision_df["label"] = pd.to_numeric(decision_df["label"], errors="coerce")
    decision_df["enter"] = go_series.to_numpy()

    tau_series = pd.to_numeric(decision_df.get("tau_used", pd.Series([], dtype=float)), errors="coerce").dropna()
    if len(tau_series):
        tau_used_val = float(tau_series.iloc[0])
    else:
        tau_used_val = float(gate_tau) if gate_tau == gate_tau else float(getattr(gate, "tau", 0.0))

    R_metric = label_r_mult
    dec = decision_df.loc[decision_df["enter"]]
    entries = int(len(dec))
    wins = int((dec["label"] == 1).sum())
    losses = int((dec["label"] == -1).sum())
    timeouts = int((dec["label"] == 0).sum())
    resolved = wins + losses

    if entries != (wins + losses + timeouts):
        _log(
            f"[sim][warn] entries({entries}) != wins+losses+timeouts({wins + losses + timeouts}); "
            "using decided_df length for entries."
        )

    total_candidates = int(len(decision_df))
    decided_cov = safe_div(entries, total_candidates)
    ppv_resolved = safe_div(wins, resolved)
    net_R = wins * R_metric - losses * 1.0
    avg_R_per_entry = safe_div(net_R, entries)
    g_ratio = (wins * R_metric) / max(1, losses)

    _log(
        "[sim] tau_used={:.4f} entries={} of {} ({:.3%}) | wins={} losses={} timeouts={} "
        "ppv_resolved={:.3f} net_R={:.2f} avg_R={:.3f} G={:.3f}".format(
            tau_used_val,
            entries,
            total_candidates,
            decided_cov,
            wins,
            losses,
            timeouts,
            ppv_resolved,
            net_R,
            avg_R_per_entry,
            g_ratio,
        )
    )

    stats_path = spath
    try:
        with stats_path.open("r", encoding="utf-8") as fh:
            stats_data = json.load(fh)
    except Exception:
        stats_data = {}

    decided_metrics = {
        "tau_used": tau_used_val,
        "entries": entries,
        "total_candidates": total_candidates,
        "total_candidates_scored": total_candidates,
        "decided_coverage": decided_cov,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "resolved": resolved,
        "ppv_resolved": ppv_resolved,
        "R_mult": R_metric,
        "net_R": net_R,
        "avg_R_per_entry": avg_R_per_entry,
        "g_ratio": g_ratio,
    }

    stats_data.update(decided_metrics)
    dump_json(stats_data, stats_path)

    coverage = decided_cov if total_candidates else (float(ev["enter"].mean()) if len(ev) else 0.0)
    empirical_ppv = ppv_resolved if resolved else float("nan")
    ppv_str = f"{empirical_ppv:.3f}" if empirical_ppv == empirical_ppv else "nan"
    _log(f"[sim] decided_cov={coverage:.3f} decided_empirical_ppv={ppv_str}")

    total = int(len(ev))
    wins_all = int((ev["outcome"] == "win").sum())
    losses_all = int((ev["outcome"] == "loss").sum())
    timeouts_all = int((ev["outcome"] == "timeout").sum())
    denom_all = max(wins_all + losses_all, 1)
    wr_all = wins_all / denom_all if denom_all else 0.0
    expR_all = wr_all * label_r_mult + (1.0 - wr_all) * (-1.0)

    trades = ev[ev["outcome"].isin(["win", "loss"])].copy()
    trades["R"] = np.where(trades["outcome"] == "win", label_r_mult, -1.0)
    trades.to_csv(tpath, index=False)

    cands_t_norm.to_csv(trpath, index=False)

    wins_fp_cols = ["ts", "side", "rule_id"] + [c for c in RULE_FINGERPRINT_FEATURES if c in ev.columns]
    ev.loc[ev["outcome"] == "win", wins_fp_cols].to_csv(fold_dir / "wins_fingerprints.csv", index=False)

    stats_existing: Dict[str, Any] = {}
    if spath.exists():
        try:
            with spath.open("r", encoding="utf-8") as f:
                stats_existing = json.load(f)
        except Exception:
            stats_existing = {}

    stats_existing.update(
        {
            "total_events": total,
            "wins_all_events": wins_all,
            "losses_all_events": losses_all,
            "timeouts_all_events": timeouts_all,
            "win_rate": wr_all,
            "expected_R_per_event": expR_all,
            "r_mult": label_r_mult,
            "gate_coverage": coverage,
        }
    )

    stats_existing.update(decided_metrics)

    dump_json(stats_existing, spath)
    _log(
        f"[{sym}] simulate: total={total}, wr={wr_all:.3f}, expR={expR_all:.2f}, "
        f"wins={wins_all}, losses={losses_all}, timeouts={timeouts_all}"
    )

    ev_g = ev[ev["enter"]].copy()

    # --- ensure scheduler-required columns exist & correct types ---
    if "end_ts" not in ev_g.columns and "outcome_ts" in ev_g.columns:
        ev_g["end_ts"] = pd.to_numeric(ev_g["outcome_ts"], errors="coerce")
    elif "end_ts" not in ev_g.columns:
        ev_g["end_ts"] = pd.Series(np.nan, index=ev_g.index, dtype="float64")
    if "pwin" not in ev_g.columns and "p_win" in ev_g.columns:
        ev_g["pwin"] = pd.to_numeric(ev_g["p_win"], errors="coerce")
    if "ts" in ev_g.columns:
        ev_g["ts"] = pd.to_numeric(ev_g["ts"], errors="coerce")
        ev_g = ev_g[ev_g["ts"].notna()].copy()
        ev_g["ts"] = ev_g["ts"].astype("int64")
    if "end_ts" in ev_g.columns:
        ev_g["end_ts"] = pd.to_numeric(ev_g["end_ts"], errors="coerce")
        ev_g = ev_g[ev_g["end_ts"].notna()].copy()
        ev_g["end_ts"] = ev_g["end_ts"].astype("int64")

    gated_count = int(ev_g.shape[0])

    # ---- progress + scheduler cfg ----
    prog_cfg = (cfg.get("progress", {}) or {}).get("sim", {}) or {}
    show_inner = bool(prog_cfg.get("inner_schedule_bar", False))
    desc_inner = str(prog_cfg.get("schedule_desc", "schedule"))
    upd_every = int(prog_cfg.get("schedule_update_every", 1000))

    sched_cfg = (cfg.get("execution_sim", {}) or {}).get("scheduler", {}) or {}
    weight_mode = str(sched_cfg.get("weight_mode", "expR"))
    r_mult = float(sched_cfg.get("r_mult", 5.0))
    timeout_sec = sched_cfg.get("timeout_sec", None)
    timeout_sec = float(timeout_sec) if timeout_sec is not None else None
    chunk_by = sched_cfg.get("chunk_by", None)
    topk_hour = sched_cfg.get("topk_per_hour", None)
    topk_hour = int(topk_hour) if topk_hour not in (None, "") else None
    enforce_xc = bool(sched_cfg.get("enforce_cross_chunk_nonoverlap", True))

    # pre-schedule snapshot to guarantee downstream artifacts exist
    ev_g.to_csv(fold_dir / "trades_gated_presched.csv", index=False)

    entry_delay_bars = int((cfg.get("execution_sim", {}) or {}).get("entry_delay_bars", 0))
    bar_ms = int(((cfg.get("data", {}) or {}).get("bar_ms", 60_000)))
    if entry_delay_bars > 0 and len(ev_g):
        shift = entry_delay_bars * bar_ms
        ev_g = ev_g.copy()
        ev_g["ts"] = ev_g["ts"] + shift
        ev_g["end_ts"] = ev_g["end_ts"] + shift

    if fold_idx is not None and len(test_months):
        progress_label = f"{desc_inner} [{sym}|Fold {fold_idx}|test={test_months[0]}]"
    elif fold_idx is not None:
        progress_label = f"{desc_inner} [{sym}|Fold {fold_idx}]"
    elif len(test_months):
        progress_label = f"{desc_inner} [{sym}|test={test_months[0]}]"
    else:
        progress_label = f"{desc_inner} [{sym}]"

    try:
        take_sched = schedule_non_overlapping(
            ev_g,
            weight_mode=weight_mode,
            r_mult=r_mult,
            show_progress=show_inner,
            progress_desc=progress_label,
            update_every=upd_every,
            timeout_sec=timeout_sec,
            chunk_by=chunk_by,
            topk_per_hour=topk_hour,
            enforce_cross_chunk_nonoverlap=enforce_xc,
        )
        scheduler_fallback = False
    except Exception as _e:
        _log(f"[{sym}] schedule_non_overlapping fallback: {_e}")
        take_sched = ev_g.copy()
        scheduler_fallback = True

    # canonical scheduled trades
    take_sched.to_csv(fold_dir / "trades_gated.csv", index=False)

    tw = (cfg.get("tripwires", {}) or {})
    ppv_lcb_min = float(tw.get("ppv_lcb_min", 0.50))
    timeout_rate_max = float(tw.get("timeout_rate_max", 0.60))
    window_days = int(tw.get("window_days", 3))
    fallback_days_max = int(tw.get("fallback_days_max", 3))

    tg = take_sched.copy()
    if "ts" in tg.columns:
        tscol = "ts"
        tg["_date"] = pd.to_datetime(tg[tscol], unit="ms", errors="coerce").dt.date
    else:
        tg["_date"] = pd.NaT

    is_win = tg.get("is_win")
    is_loss = tg.get("is_loss")
    is_to = tg.get("is_to")
    if is_win is None or is_loss is None:
        oc = tg.get("outcome")
        if oc is not None:
            oc = oc.fillna("")
            is_win = (oc == "win").astype(int)
            is_loss = (oc == "loss").astype(int)
            if "timeout" in set(oc.unique()):
                is_to = (oc == "timeout").astype(int)
            else:
                is_to = pd.Series([0] * len(tg), index=tg.index, dtype=int)
        else:
            zeros = pd.Series([0] * len(tg), index=tg.index, dtype=int)
            is_win = zeros.copy()
            is_loss = zeros.copy()
            is_to = zeros.copy()

    if not isinstance(is_win, pd.Series):
        is_win = pd.Series(is_win, index=tg.index)
    if not isinstance(is_loss, pd.Series):
        is_loss = pd.Series(is_loss, index=tg.index)
    if not isinstance(is_to, pd.Series):
        is_to = pd.Series(is_to, index=tg.index)

    tg["is_win"] = pd.to_numeric(is_win, errors="coerce").fillna(0).astype(int)
    tg["is_loss"] = pd.to_numeric(is_loss, errors="coerce").fillna(0).astype(int)
    tg["is_to"] = pd.to_numeric(is_to, errors="coerce").fillna(0).astype(int)

    daily = tg.groupby("_date").agg(
        wins=("is_win", "sum") if "is_win" in tg.columns else ("_date", "size"),
        losses=("is_loss", "sum") if "is_loss" in tg.columns else ("_date", "size"),
        timeouts=("is_to", "sum") if "is_to" in tg.columns else ("_date", "size"),
        scheduled=("_date", "size"),
    ).reset_index()

    daily["wins_roll"] = daily["wins"].rolling(window_days, min_periods=1).sum()
    daily["losses_roll"] = daily["losses"].rolling(window_days, min_periods=1).sum()
    daily["resolved_roll"] = daily["wins_roll"] + daily["losses_roll"]
    daily["ppv_lcb_roll"] = daily.apply(
        lambda r: wilson_lcb(int(r["wins_roll"]), int(r["resolved_roll"])) if r["resolved_roll"] > 0 else 0.0,
        axis=1,
    )
    daily["timeout_rate_roll"] = (
        daily["timeouts"].rolling(window_days, min_periods=1).sum()
        / daily["scheduled"].rolling(window_days, min_periods=1).sum().clip(lower=1)
    )

    ppv_breach_days = [str(d) for d, v in zip(daily["_date"], daily["ppv_lcb_roll"]) if v < ppv_lcb_min]
    to_breach_days = [str(d) for d, v in zip(daily["_date"], daily["timeout_rate_roll"]) if v > timeout_rate_max]

    tripwire = {
        "ppv_lcb_min": ppv_lcb_min,
        "timeout_rate_max": timeout_rate_max,
        "window_days": window_days,
        "fallback_days_max": fallback_days_max,
        "ppv_lcb_roll_min": float(daily["ppv_lcb_roll"].min() if len(daily) else 0.0),
        "timeout_rate_roll_max": float(daily["timeout_rate_roll"].max() if len(daily) else 0.0),
        "ppv_breach_days": ppv_breach_days,
        "timeout_breach_days": to_breach_days,
        "scheduler_fallback_used": bool(scheduler_fallback),
    }

    with open(fold_dir / "tripwire_status.json", "w") as fh:
        json.dump(tripwire, fh, indent=2)

    if ppv_breach_days or to_breach_days:
        _log(f"[{sym}] TRIPWIRE breach: {tripwire}")

    _log(f"[SIM] coverage={coverage:.3f} selected={gated_count} scheduled={len(take_sched)}")

    outcome_series = take_sched.get("outcome")
    if outcome_series is None:
        outcome_series = pd.Series(np.full(len(take_sched), "", dtype=object), index=take_sched.index)
    g_wins = int((outcome_series == "win").sum())
    g_losses = int((outcome_series == "loss").sum())
    g_timeouts = int((outcome_series == "timeout").sum())
    g_denom = max(g_wins + g_losses, 1)
    g_wr = g_wins / g_denom if g_denom else 0.0
    g_expR = g_wr * label_r_mult + (1.0 - g_wr) * (-1.0)

    wins_fp_cols_g = ["ts", "side"] + [c for c in RULE_FINGERPRINT_FEATURES if c in take_sched.columns]
    take_sched.loc[outcome_series == "win", wins_fp_cols_g].to_csv(
        fold_dir / "wins_fingerprints_gated.csv",
        index=False,
    )

    expR_series = take_sched.get("expR", pd.Series([], dtype=float))
    expR_series = pd.to_numeric(expR_series, errors="coerce") if isinstance(expR_series, pd.Series) else pd.Series(expR_series)
    expected_R_per_event = float(expR_series.fillna(0.0).mean()) if len(expR_series) else 0.0
    outcome_non_empty = outcome_series.dropna()
    outcome_non_empty = outcome_non_empty[outcome_non_empty != ""]
    denom_outcomes = max(1, int(len(outcome_non_empty)))

    write_json(
        fold_dir / "stats_gated.json",
        {
            "total_events_gated": int(len(take_sched)),
            "wins": g_wins,
            "losses": g_losses,
            "timeouts": g_timeouts,
            "win_rate": float(g_wins / denom_outcomes) if denom_outcomes else 0.0,
            "expected_R_per_event": expected_R_per_event,
        "r_mult": r_mult,
            "scheduler_fallback": scheduler_fallback,
        },
    )

    _log(
        f"[{sym}] simulate[GATED]: kept={len(ev_g)} scheduled={len(take_sched)} wr={g_wr:.3f} expR={g_expR:.2f}"
    )


# ------------------------ orchestrator ------------------------


def run_walkforward(cfg_path: str, force: bool, clean: bool, only: Optional[str], from_stage: Optional[str], until_stage: Optional[str]) -> None:


    cfg = load_cfg(cfg_path)


    out_root = Path(cfg.get("io", {}).get("outputs_root", "outputs"))


    ensure_dir(out_root)


    months = month_list(cfg["data"]["months"]["start"], cfg["data"]["months"]["end"])


    train = int(cfg["walkforward"].get("train_months", cfg["data"]["months"].get("train_span", 3)))


    test  = int(cfg["walkforward"].get("test_months",  cfg["data"]["months"].get("test_span", 1)))


    step  = int(cfg["walkforward"].get("step_months",  cfg["data"]["months"].get("step", 1)))


    folds = make_folds(months, train, test, step)


    symbols: List[str] = cfg["data"]["symbols"]


    stages = stages_to_run(only, from_stage, until_stage)


    _log("Preflight OK")


    _log(f"Symbols: {symbols}")


    _log(f"Months:  {months}")


    _log(f"Folds:   {len(folds)} (train={train}, test={test}, step={step})")


    progress_on = bool(cfg.get("io", {}).get("progress", True))


    skip_set = set((cfg.get("walkforward", {}) or {}).get("skip_folds", []))


    for sym in symbols:


        fold_iter = _pbar(list(enumerate(folds)), desc=f"{sym} folds", enabled=progress_on, total=len(folds))


        for i, (tr_m, te_m) in fold_iter:


            if i in skip_set:
                _log(f"[{sym}] SKIP fold {i}")
                continue


            fold_dir = out_root / sym / f"fold_{i}"


            if clean and fold_dir.exists():


                shutil.rmtree(fold_dir)


            ensure_dir(fold_dir)


            _log(f"\n[{sym} | Fold {i}] train={tr_m} -> test={te_m}")


            if "features" in stages:


                try:


                    stage_features(sym, tr_m, te_m, cfg, fold_dir, force)


                except Exception as e:


                    rep = exception_to_report("features", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


            if "tick_labeling" in stages:


                try:


                    stage_tick_labeling(sym, tr_m, te_m, cfg, fold_dir, force)


                except Exception as e:


                    rep = exception_to_report("tick_labeling", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


            if "mining" in stages:


                try:


                    stage_mining(sym, tr_m, te_m, cfg, fold_dir, force)


                except Exception as e:


                    rep = exception_to_report("mining", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


            if "simulate" in stages:


                try:


                    stage_simulate(sym, tr_m, te_m, cfg, fold_dir, force, fold_idx=i)


                except Exception as e:


                    rep = exception_to_report("simulate", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


# ------------------------ main ------------------------


def main():


    args = parse_args()


    if args.mode == "preflight":


        cfg = load_cfg(args.config)


        _log("Preflight OK")


        _log(f"Symbols: {cfg['data']['symbols']}")


        _log(f"Months:  {cfg['data']['months']['start']}..{cfg['data']['months']['end']}")


        return


    if args.mode == "walkforward":


        run_walkforward(


            args.config,


            force=args.force,


            clean=args.clean,


            only=args.only,


            from_stage=args.from_stage,


            until_stage=args.until_stage,


        )


        _log("Walkforward DONE")


        return


    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":


    main()


