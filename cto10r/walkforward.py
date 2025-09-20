# cto10r/walkforward.py


from __future__ import annotations


import argparse


import json


import shutil


from dataclasses import dataclass


from pathlib import Path


from typing import Any, Dict, Iterable, List, Optional, Tuple


import numpy as np


import pandas as pd


import yaml
from tqdm.auto import tqdm


# --- project imports that exist in your repo ---


from .io_utils import ensure_dir, write_json, skip_if_exists


from .util import exception_to_report, infer_bar_seconds


from .bars import load_bars_any, build_features, fit_percentiles, apply_percentiles, add_age_features, add_nonlinear_features


from .candidates import build_candidates_router


from .ticks import iter_ticks_files, stream_ticks_window, label_events_from_ticks


from .mining import mine_rules, prepare_literal_buckets
from .ml import train_ml_gating, score_ml, schedule_non_overlapping


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


    res = mine_rules(cands_tr, events_tr, {"rules": rules_cfg, "out_dir": str(fold_dir), "features": cfg.get("features", {}), "progress": progress_on})


    promoted = res.get("promoted", [])


    _log(f"[{sym}] mining: rows={len(events_tr)} promoted={len(promoted)}")

    # Train ML gating on TRAIN
    ml_cfg = cfg.get("ml_gating", {})
    if bool(ml_cfg.get("enabled", False)):
        _log(f"[{sym}] ml_gating: training {ml_cfg.get('model','logreg')}...")
        try:
            train_ml_gating(cands_tr, events_tr, train_months, {"ml_gating": ml_cfg}, fold_dir)
            # log ML meta summary
            try:
                art = Path(fold_dir) / "artifacts"
                with open(art / "ml_meta.json","r",encoding="utf-8") as f:
                    meta = json.load(f)
                _log(f"[{sym}] ml_gating: model={meta.get('model')} params={meta.get('params')} AP_val={meta.get('ap_val'):.3f} tau={meta.get('tau'):.4f} LCB@tau={meta.get('achieved_lcb'):.3f} cov={meta.get('coverage_val'):.3f}")
            except Exception:
                pass
        except Exception as e:
            _log(f"[{sym}] ml_gating: training failed: {e}")


def stage_simulate(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], fold_dir: Path, force: bool) -> None:


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


    r_mult = float(cfg["labels"].get("r_mult", 10.0))


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
    blocklist = []

    gpath = fold_dir / "artifacts" / "gating.json"

    if gpath.exists():

        with open(gpath, "r") as f:

            data = json.load(f)
            rules = data.get("promoted", [])
            blocklist = data.get("blocklist", [])

    ev["rule_id"] = assign_best_rule_id(ev, rules) if rules else None


    ev = ev.sort_values("ts").reset_index(drop=True)


    total = int(len(ev))


    wins = int((ev["outcome"] == "win").sum())


    losses = int((ev["outcome"] == "loss").sum())


    timeouts = int((ev["outcome"] == "timeout").sum())


    denom = max(wins + losses, 1)


    wr = wins / denom if denom else 0.0


    expR = wr * r_mult + (1.0 - wr) * (-1.0)


    trades = ev[ev["outcome"].isin(["win", "loss"])].copy()


    trades["R"] = np.where(trades["outcome"] == "win", r_mult, -1.0)


    trades.to_csv(tpath, index=False)


    cands_t_norm.to_csv(trpath, index=False)


    wins_fp_cols = ["ts", "side", "rule_id"] + [c for c in RULE_FINGERPRINT_FEATURES if c in ev.columns]


    ev.loc[ev["outcome"] == "win", wins_fp_cols].to_csv(fold_dir / "wins_fingerprints.csv", index=False)


    stats = {


        "total_events": total,


        "wins": wins,


        "losses": losses,


        "timeouts": timeouts,


        "win_rate": wr,


        "expected_R_per_event": expR,


        "r_mult": r_mult,


    }


    write_json(spath, stats)


    _log(f"[{sym}] simulate: total={total}, wr={wr:.3f}, expR={expR:.2f}, wins={wins}, losses={losses}, timeouts={timeouts}")


    sim_cfg = cfg.get("execution_sim", {})
    ml_cfg  = cfg.get("ml_gating", {})

    # Apply loser blocklist first, if configured
    if bool(sim_cfg.get("use_blocklist", False)) and blocklist:
        def _row_matches_any(row):
            for r in blocklist:
                conds = r.get("conds", [])
                if conds and rule_hit_row(row, conds):
                    return True
            return False
        if not ev.empty:
            mask_block = ev.apply(_row_matches_any, axis=1)
            blocked = int(mask_block.sum())
            if blocked > 0:
                _log(f"[{sym}] simulate: blocklist filtered {blocked} events")
            ev = ev.loc[~mask_block].copy()

    # ML gating
    if bool(ml_cfg.get("enabled", False)) and len(cands_t_norm) > 0:
        try:
            p = score_ml(cands_t_norm, fold_dir, ml_cfg)
            cands_t_norm = cands_t_norm.copy()
            cands_t_norm["pwin"] = p
            from pathlib import Path as _P
            with open(_P(fold_dir)/"artifacts"/"ml_meta.json","r",encoding="utf-8") as f:
                meta = json.load(f)
            tau = float(meta.get("tau", 0.5))
            cands_t_norm["ml_gate"] = (cands_t_norm["pwin"] >= tau).astype(bool)
            ev = ev.merge(cands_t_norm[["ts","pwin","ml_gate"]], on="ts", how="left")
            ev["pwin"] = pd.to_numeric(ev["pwin"], errors="coerce")
            ev["ml_gate"] = ev["ml_gate"].fillna(False)
        except Exception as e:
            _log(f"[{sym}] simulate: ML scoring failed: {e}")

    # Winner voting (k-of-n) if rules used
    if bool(sim_cfg.get("use_rules", False)) and rules:
        thr = float(sim_cfg.get("min_rule_precision_lcb", 0.0))
        sup_min = int(sim_cfg.get("min_rule_support", 0))
        good_rules = [r for r in rules if r.get("support_n", 0) >= sup_min and r.get("precision_lcb", 0.0) >= thr]
        min_votes = int(sim_cfg.get("min_winner_votes", 1))
        def _votes(row):
            v = 0
            for r in good_rules:
                conds = r.get("conds", [])
                if conds and rule_hit_row(row, conds):
                    v += 1
            return v
        if not ev.empty:
            ev["winner_votes"] = ev.apply(_votes, axis=1).astype(int)
            mask_motif = (ev["winner_votes"] >= min_votes).to_numpy()
        else:
            mask_motif = np.zeros(len(ev), dtype=bool)
    else:
        mask_motif = np.ones(len(ev), dtype=bool)

    # ML mask
    if "ml_gate" in ev.columns:
        mask_ml = ev["ml_gate"].to_numpy()
    else:
        mask_ml = np.ones(len(ev), dtype=bool)

    # Regime mask: any qbin_agebin_* >= regime_min_age_bin
    mask_reg = np.ones(len(ev), dtype=bool)
    if bool(ml_cfg.get("regime_gate", False)) and len(ev) > 0:
        thr_age = int(ml_cfg.get("regime_min_age_bin", 4))
        cols_age = [c for c in ev.columns if c.startswith("qbin_agebin_")]
        if cols_age:
            m = np.zeros(len(ev), dtype=bool)
            for c in cols_age:
                try:
                    m |= (pd.to_numeric(ev[c], errors="coerce") >= thr_age).to_numpy()
                except Exception:
                    continue
            mask_reg = m

    combine = str(ml_cfg.get("combine_with_motifs", "and")).lower()
    if combine == "and":
        keep_mask = mask_motif & mask_ml & mask_reg
    elif combine == "or":
        keep_mask = (mask_motif | mask_ml) & mask_reg
    elif combine == "ml_only":
        keep_mask = mask_ml & mask_reg
    elif combine == "motifs_only":
        keep_mask = mask_motif & mask_reg
    else:
        keep_mask = mask_motif & mask_ml & mask_reg

    ev_g = ev.loc[keep_mask].copy()

    g_wins = int((take_sched["outcome"] == "win").sum())
    g_losses = int((take_sched["outcome"] == "loss").sum())
    g_timeouts = int((take_sched["outcome"] == "timeout").sum())
    g_denom = max(g_wins + g_losses, 1)
    g_wr = g_wins / g_denom if g_denom else 0.0
    g_expR = g_wr * r_mult + (1.0 - g_wr) * (-1.0)

    # Scheduler
    sched_cfg = sim_cfg.get("scheduler", {"enabled": True, "weight": "expR"})
    if bool(sched_cfg.get("enabled", True)):
        take_sched = schedule_non_overlapping(ev_g, weight_mode=str(sched_cfg.get("weight","expR")).lower(), r_mult=r_mult)
    else:
        take_sched = ev_g

    trades_g = take_sched[take_sched["outcome"].isin(["win", "loss"])].copy()
    trades_g["R"] = np.where(trades_g["outcome"] == "win", r_mult, -1.0)
    trades_g.to_csv(fold_dir / "trades_gated.csv", index=False)

    wins_fp_cols_g = ["ts", "side"] + [c for c in RULE_FINGERPRINT_FEATURES if c in take_sched.columns]
    take_sched.loc[take_sched["outcome"] == "win", wins_fp_cols_g].to_csv(fold_dir / "wins_fingerprints_gated.csv", index=False)

    write_json(
        fold_dir / "stats_gated.json",
        {
            "total_events_gated": int(len(take_sched)),
            "wins": g_wins,
            "losses": g_losses,
            "timeouts": g_timeouts,
            "win_rate": g_wr,
            "expected_R_per_event": g_expR,
            "r_mult": r_mult,
            "rules_used": int(len(rules)),
        },
    )

    _log(f"[{sym}] simulate[GATED]: kept={len(ev_g)} scheduled={len(take_sched)} wr={g_wr:.3f} expR={g_expR:.2f} combine={combine}")


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


    for sym in symbols:


        fold_iter = _pbar(list(enumerate(folds)), desc=f"{sym} folds", enabled=progress_on, total=len(folds))


        for i, (tr_m, te_m) in fold_iter:


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


                    stage_simulate(sym, tr_m, te_m, cfg, fold_dir, force)


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


