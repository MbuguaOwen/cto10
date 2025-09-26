# cto10r/walkforward.py


from __future__ import annotations


import argparse


import hashlib
import json
import os
import random
import shutil
import copy
import sys


from dataclasses import dataclass
import math


from pathlib import Path


from typing import Any, Dict, Iterable, List, Optional, Tuple


import numpy as np


import pandas as pd


import yaml
from tqdm.auto import tqdm


# --- project imports that exist in your repo ---


from .io_utils import ensure_dir, write_json, skip_if_exists


from .util import (
    exception_to_report,
    infer_bar_seconds,
    safe_div,
    dump_json,
    normalize_cands_schema,
    normalize_events_schema,
    ensure_parquet_engine,
    wilson_lcb,
)


from .bars import load_bars_any, build_features, fit_percentiles, apply_percentiles, add_age_features, add_nonlinear_features


from .candidates import build_candidates_router


from .ticks import iter_ticks_files, stream_ticks_window, label_events_from_ticks


from .mining import mine_rules, prepare_literal_buckets, match_rules_vectorized
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
    "close_to_hi_atr",  # directional, replaces close_to_ext_atr
    "close_to_lo_atr",  # directional, replaces close_to_ext_atr
    "atr_p", "dcw_p",
    "eta",
    "ym",
]


def _write_empty_trades_csv(path):
    import pandas as pd
    cols = [
        "ts",
        "side",
        "entry",
        "tp",
        "sl",
        "exit_ts",
        "outcome",
        "r",
        "preempted",
        "tau_used",
        "p_win",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


ALL_STAGES = ["features", "tick_labeling", "mining", "simulate"]


# ------------------------ fingerprints ------------------------


def _sha256_path(p: Path) -> str:
    try:
        with open(p, 'rb') as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return ''


def _sha256_text(txt: str) -> str:
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


def _write_fingerprint(out_dir: Path, stage: str, cfg_path: Path, extra: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = {
        'stage': stage,
        'python': sys.version,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'cfg_path': str(cfg_path),
        'cfg_sha256': _sha256_path(cfg_path),
        'code': {
            'ticks.py': _sha256_path(Path(__file__).parent / 'ticks.py'),
            'walkforward.py': _sha256_path(Path(__file__)),
            'util.py': _sha256_path(Path(__file__).parent / 'util.py'),
        },
    }
    if extra:
        fp.update(extra)
    (out_dir / f'run_fingerprint_{stage}.json').write_text(json.dumps(fp, indent=2))


def _describe_paths(paths: Iterable[Path | str]) -> List[Dict[str, Any]]:
    info: List[Dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        entry: Dict[str, Any] = {'path': str(p)}
        try:
            stat = p.stat()
            entry.update({'size': stat.st_size, 'mtime': stat.st_mtime, 'sha256': _sha256_path(p)})
        except Exception:
            pass
        info.append(entry)
    return info


def _seed_determinism() -> None:
    os.environ.setdefault('PYTHONHASHSEED', '0')
    random.seed(42)
    np.random.seed(42)


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
    # Vectorized first-hit matching using rule conditions
    if df.empty or not rules:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    ser = match_rules_vectorized(df, rules, lit_prefix="B_")
    return ser


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


    try:


        with open(path, "r", encoding="utf-8") as f:


            return yaml.safe_load(f)


    except UnicodeDecodeError:


        # Some editors save UTF-8 with BOM; yaml can read it fine if we decode with utf-8-sig
        with open(path, "r", encoding="utf-8-sig") as f:


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
    files_used: List[str] = []


    for p in _pbar(paths, desc=f"{sym} bars", enabled=progress_on, total=len(paths)):


        if not p.exists():


            continue


        df = load_bars_any(p, schema=bar_schema)


        dfs.append(df)
        files_used.append(str(p))


    if not dfs:


        bars = pd.DataFrame(columns=["ts","open","high","low","close","ym"])
        bars.attrs["files_used"] = files_used
        return bars


    bars = pd.concat(dfs, ignore_index=True).sort_values("ts")
    bars.attrs["files_used"] = files_used


    # derive ym (YYYY-MM) from timestamp (ms)


    ts = pd.to_datetime(bars["ts"], unit="ms", utc=True)


    bars["ym"] = ts.dt.strftime("%Y-%m")


    return bars


# ------------------------ stages ------------------------


def stage_features(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], cfg_path: Path, fold_dir: Path, force: bool) -> None:


    _log(f"[{sym}] features: start train={train_months} test={test_months}")


    ensure_dir(fold_dir)


    artifacts_dir = fold_dir / "artifacts"
    ensure_dir(artifacts_dir)

    cfg_echo = fold_dir / "cfg_echo.json"


    if not cfg_echo.exists():


        write_json(cfg_echo, cfg)


    cpath = fold_dir / "candidates.parquet"


    ensure_parquet_engine()


    if skip_if_exists(cpath, force):


        _log(f"[{sym}] SKIP features (exists)")


        return


    data_cfg = cfg["data"]


    bar_schema = data_cfg.get("bar_schema", "auto")


    progress_on = bool(data_cfg.get("progress", cfg.get("io", {}).get("progress", True)))


    months_all = list(dict.fromkeys(train_months + test_months))


    bars = _load_bars_months(sym, months_all, data_cfg, bar_schema, progress_on)


    if bars.empty:


        empty = normalize_cands_schema(pd.DataFrame())
        empty.to_parquet(cpath, index=False)

        bar_files_info = _describe_paths(bars.attrs.get("files_used", []))
        _write_fingerprint(
            artifacts_dir,
            "features",
            cfg_path,
            {
                "train_months": list(train_months),
                "test_months": list(test_months),
                "bars_rows": 0,
                "candidates_rows": 0,
                "bar_files": bar_files_info,
            },
        )

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

    # Compat: derive close_to_ext_atr if someone expects it
    if "close_to_ext_atr" not in feat_all.columns:
        if {"t_240", "close_to_hi_atr", "close_to_lo_atr"}.issubset(feat_all.columns):
            t_240_numeric = pd.to_numeric(feat_all["t_240"], errors="coerce")
            hi_atr = pd.to_numeric(feat_all["close_to_hi_atr"], errors="coerce")
            lo_atr = pd.to_numeric(feat_all["close_to_lo_atr"], errors="coerce")
            feat_all["close_to_ext_atr"] = np.where(t_240_numeric >= 0, hi_atr, lo_atr)

    cands = build_candidates_router(feat_all, cfg)


    cands = cands.sort_values("ts").reset_index(drop=True)
    # Drop duplicate columns to keep PyArrow happy
    cands = cands.loc[:, ~cands.columns.duplicated()]
    cands = normalize_cands_schema(cands)

    bar_files_info = _describe_paths(bars.attrs.get("files_used", []))
    _write_fingerprint(
        artifacts_dir,
        "features",
        cfg_path,
        {
            "train_months": list(train_months),
            "test_months": list(test_months),
            "bars_rows": int(len(bars)),
            "candidates_rows": int(len(cands)),
            "bar_files": bar_files_info,
        },
    )

    cands.to_parquet(cpath, index=False)


    _log(f"[{sym}] features: candidates={len(cands)}")


def stage_tick_labeling(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], cfg_path: Path, fold_dir: Path, force: bool) -> None:


    _log(f"[{sym}] tick_labeling: start")


    artifacts_dir = fold_dir / "artifacts"
    ensure_dir(artifacts_dir)

    epath = fold_dir / "events.parquet"


    ensure_parquet_engine()


    if skip_if_exists(epath, force):


        _log(f"[{sym}] SKIP tick_labeling (exists)")


        return


    # Friendly guard: require features stage to exist
    cpath = fold_dir / "candidates.parquet"
    if not cpath.exists():
        raise FileNotFoundError(
            f"{cpath} not found. Run features stage first:\n"
            f"  python -m cto10r --config {cfg_path} --mode walkforward --only features --force"
        )
    cands = pd.read_parquet(cpath)
    # Fast labeling cap for debugging (optional)
    import os as _os
    cap = int(_os.getenv("LABEL_MAX", "0") or "0")
    if cap > 0:
        cands = cands.sort_values("ts").head(cap).copy()
        _log(f"[debug] LABEL_MAX={cap} → cands={len(cands)}")

    # Volatility prefilter intentionally disabled (vol_prefilter: null)
    # (prefilter disabled)


    if cands.empty:


        empty_events = pd.DataFrame(columns=["ts", "side", "entry", "level", "risk_dist", "outcome"])
        empty_events = normalize_events_schema(empty_events)
        empty_events.attrs["no_ticks_windows"] = 0
        empty_events.to_parquet(epath, index=False)


        _log(f"[{sym}] events: 0 (no candidates)")


        _write_fingerprint(
            artifacts_dir,
            "tick_labeling",
            cfg_path,
            {
                "train_months": list(train_months),
                "test_months": list(test_months),
                "events": 0,
                "wins": 0,
                "losses": 0,
                "timeouts": 0,
                "preempted": 0,
                "no_ticks_windows": 0,
                "tick_files": tick_file_info,
            },
        )


        return


    # ---- BEGIN: unit-safe per-side best-in-window quick dedupe (2h) ----
    import numpy as np

    def _eta_proxy_from_atr_p(series_atr_p: pd.Series, labels_cfg: dict) -> pd.Series:
        rules = (labels_cfg.get("eta_by_atr_p") or [])
        if not len(rules):
            return pd.Series(np.nan, index=series_atr_p.index)
        def map_one(p):
            p = float(p) if pd.notna(p) else 50.0
            for lo, hi, val in rules:
                if p >= lo and p < hi:
                    return float(val)
            return float(rules[-1][2])
        eta = series_atr_p.map(map_one)
        cap = labels_cfg.get("eta_cap", {})
        return eta.clip(lower=float(cap.get("min", eta.min() if len(eta) else 0.0)),
                        upper=float(cap.get("max", eta.max() if len(eta) else 1e9)))

    def _detect_ts_unit(tser: pd.Series) -> str:
        """Return 'sec'|'ms'|'ns' by inspecting positive deltas only."""
        t = pd.to_numeric(tser, errors="coerce").astype("int64")
        dt = t.sort_values().diff()
        dt = dt[dt > 0]                         # ignore zeros/dupes (skewed median)
        q = float(dt.median()) if len(dt) else 60.0
        # thresholds: ~1s, ~1e3 ms, ~1e9 ns
        if q >= 1e10:  # ~>10 billion ns ~ 10s → ns-level stamps
            return "ns"
        if q >= 1e4:   # ~>10k ms → ms-level stamps
            return "ms"
        return "sec"

    def _qh_in_unit(hours: float, unit: str) -> int:
        if unit == "sec": return int(hours * 3_600)
        if unit == "ms":  return int(hours * 3_600_000)
        return int(hours * 3_600_000_000_000)  # ns

    def _fmt_delta(d: int, unit: str) -> str:
        denom = {"sec": 60, "ms": 60_000, "ns": 60_000_000_000}
        mins = d / denom[unit]
        if mins < 120:
            return f"{mins:.0f}m"
        return f"{mins/60:.1f}h"

    def dedupe_best_in_window(cands_df: pd.DataFrame, cfg: dict, sym: str) -> pd.DataFrame:
        if cands_df.empty:
            _log(f"[{sym}] dedupe: no candidates")
            return cands_df

        c = cands_df.copy()
        c["side"] = c["side"].astype(str).str.strip().str.lower()
        c["ts"]   = pd.to_numeric(c["ts"], errors="coerce").astype("int64")

        labels_cfg = cfg.get("labels", {}) or {}
        dcfg = cfg.get("dedupe", {}) or {}
        scope  = str(dcfg.get("scope", "side")).lower()          # 'side' or 'symbol'
        rankby = str(dcfg.get("rank_by", "eta_proxy_then_atr")).lower()
        qh_hours = float(labels_cfg.get("quick_ignition_hours", 2.0))

        unit = _detect_ts_unit(c["ts"])
        qh   = _qh_in_unit(qh_hours, unit)

        # ranking: lowest eta proxy (closest TP), then highest atr_p, then earliest ts
        eta_proxy = c["eta"].astype(float) if ("eta" in c.columns and rankby.startswith("eta")) \
                    else _eta_proxy_from_atr_p(c.get("atr_p", pd.Series(index=c.index)), labels_cfg)
        atrp = c.get("atr_p", pd.Series(-np.inf, index=c.index)).astype(float)

        c["_rank_eta"] = eta_proxy
        c["_rank_atr"] = atrp

        keep_idx = []
        groups = [("symbol", c)] if scope == "symbol" else list(c.groupby("side", observed=True))
        for gname, g in groups:
            g = g.sort_values(["ts"]).reset_index()
            i, n = 0, len(g)
            while i < n:
                start_ts = int(g.loc[i, "ts"])
                # window = [start_ts, start_ts + qh)
                j = i
                limit = start_ts + qh
                while j < n and int(g.loc[j, "ts"]) < limit:
                    j += 1
                w = g.loc[i:j-1] if j > i else g.loc[i:i]
                w = w.sort_values(by=["_rank_eta", "_rank_atr", "ts"],
                                  ascending=[True, False, True])
                best = w.iloc[0]
                keep_idx.append(int(best["index"]))
                # next window starts at best.ts + qh
                next_limit = int(best["ts"]) + qh
                while j < n and int(g.loc[j, "ts"]) < next_limit:
                    j += 1
                i = j

        kept = cands_df.loc[sorted(set(keep_idx))].sort_values("ts").reset_index(drop=True)

        # spacing sanity per scope
        stats = []
        if scope == "symbol":
            dt = pd.to_numeric(kept["ts"], errors="coerce").astype("int64").diff().dropna()
            if len(dt): stats.append(f"minΔ={_fmt_delta(int(dt.min()), unit)}")
        else:
            for s, gg in kept.groupby(kept["side"], observed=True):
                dt = pd.to_numeric(gg["ts"], errors="coerce").astype("int64").diff().dropna()
                if len(dt): stats.append(f"{s}:minΔ={_fmt_delta(int(dt.min()), unit)}")
        stats_str = " ".join(stats)

        kept_ratio = len(kept)/max(1,len(cands_df))
        _log(f"[{sym}] dedupe[quick={qh_hours:.0f}h, scope={scope}, rank=eta→atr, unit={unit}]: "
             f"kept {len(kept)}/{len(cands_df)} ({kept_ratio:.1%}); {stats_str}")
        # Soft sanity: warn if count is implausibly high for 2h windows in 1 month
        if kept_ratio > 0.5 and len(kept) > 2000:
            _log(f"[{sym}] WARNING: dedupe kept unusually many; check ts units & window logic.")
        return kept

    # ensure we print the prefilter line ONCE
    _log(f"[{sym}] prefilter: disabled")

    _before = len(cands)
    cands = dedupe_best_in_window(cands, cfg, sym)
    # ---- END: unit-safe per-side best-in-window quick dedupe (2h) ----


    _log(f"[{sym}] tick_labeling: candidates={len(cands)}")

    tick_file_info: List[Dict[str, Any]] = []
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
    # Policy and epsilon for tick labeling
    non_overlap_policy = str(labels_cfg.get("non_overlap_policy", "quick"))
    intra_eps_ms = int(labels_cfg.get("intra_bar_epsilon_ms", 1))
    min_risk_atr_frac = float(labels_cfg.get("min_risk_atr_frac", 0.001))
    no_ticks_policy = str(labels_cfg.get("no_ticks_policy", "timeout"))


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
    tick_file_info = []
    for ym, path_obj in tick_files:
        entry = {"ym": ym, "path": str(path_obj)}
        try:
            stat = Path(path_obj).stat()
            entry.update({"size": stat.st_size, "mtime": stat.st_mtime, "sha256": _sha256_path(Path(path_obj))})
        except Exception:
            pass
        tick_file_info.append(entry)
    print(f"[ticks] files: {len(tick_files)} files for {sym} {test_months}")

    progress_on = bool(cfg.get("io", {}).get("progress", True))


    if not tick_files:


        ev = cands.copy()


        ev["outcome"] = "timeout"


        ev["outcome_ts"] = ev["ts"]


        ev["r1_ts"] = np.nan


        ev["tp_ts"] = np.nan


        ev = normalize_events_schema(ev)
        ev.attrs["no_ticks_windows"] = len(ev)
        ev.to_parquet(epath, index=False)


        _log(f"[{sym}] events: {len(ev)} (wins=0, losses=0, timeouts={len(ev)})")


        _write_fingerprint(
            artifacts_dir,
            "tick_labeling",
            cfg_path,
            {
                "train_months": list(train_months),
                "test_months": list(test_months),
                "events": int(len(ev)),
                "wins": 0,
                "losses": 0,
                "timeouts": int(len(ev)),
                "preempted": 0,
                "no_ticks_windows": int(len(ev)),
                "tick_files": tick_file_info,
            },
        )


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
        policy=non_overlap_policy,
        intra_eps_ms=intra_eps_ms,
        min_risk_frac=min_risk_atr_frac,
        no_ticks_policy=no_ticks_policy,


    )


    ev = ev.sort_values("ts").reset_index(drop=True)
    ev = normalize_events_schema(ev)

    ev.to_parquet(epath, index=False)


    wins = int((ev["outcome"] == "win").sum())


    losses = int((ev["outcome"] == "loss").sum())


    timeouts = int((ev["outcome"] == "timeout").sum())


    preempted = int(ev.get("preempted", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum()) if len(ev) else 0
    timeouts_nonpreempt = max(timeouts - preempted, 0)
    _log(f"[{sym}] events: {len(ev)} (wins={wins}, losses={losses}, timeouts={timeouts})")
    _log(f"[{sym}] events detail: preempted={preempted}, timeouts_nonpreempt={timeouts_nonpreempt}")

    no_ticks_windows = int(ev.attrs.get("no_ticks_windows", 0))
    # Tripwires for data quality
    bad_ticks_thr = 0.10  # 10% default
    if no_ticks_windows / max(len(ev), 1) > bad_ticks_thr:
        _log(f"[{sym}] WARNING: no-ticks windows > {bad_ticks_thr*100:.0f}% — data gaps may bias outcomes")
    fallback_risk_count = int(ev.attrs.get("fallback_risk_count", 0))
    if fallback_risk_count > 0:
        _log(f"[{sym}] INFO: applied ATR-based risk fallback {fallback_risk_count} times")
    _write_fingerprint(
        artifacts_dir,
        "tick_labeling",
        cfg_path,
        {
            "train_months": list(train_months),
            "test_months": list(test_months),
            "events": int(len(ev)),
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "preempted": preempted,
            "timeouts_nonpreempt": timeouts_nonpreempt,
            "no_ticks_windows": no_ticks_windows,
            "tick_files": tick_file_info,
        },
    )


def stage_mining(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], cfg_path: Path, fold_dir: Path, force: bool) -> None:


    _log(f"[{sym}] mining: start train={train_months}")


    apath = fold_dir / "artifacts" / "gating.json"


    if skip_if_exists(apath, force):


        _log(f"[{sym}] SKIP mining (exists)")


        return


    ensure_dir(fold_dir / "artifacts")


    # Friendly guard: require features stage to exist
    cpath = fold_dir / "candidates.parquet"
    if not cpath.exists():
        raise FileNotFoundError(
            f"{cpath} not found. Run features stage first:\n"
            f"  python -m cto10r --config {cfg_path} --mode walkforward --only features --force"
        )
    cands = pd.read_parquet(cpath)


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

    if "preempted" in events_tr.columns:
        events_tr = events_tr[~events_tr["preempted"].astype(bool)].copy()


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
    gate_summary = {
        "tau": float("nan"),
        "ppv": float("nan"),
        "ppv_lcb": float("nan"),
        "coverage": float("nan"),
        "loser_rules": 0,
    }
    gate_summary_path = artifacts_dir / "gate_summary.json"
    diag: Dict[str, Any] = {}

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
                f"[{sym}] gate: tau={diag['tau']:.4f} PPV={diag['ppv']:.3f} LCB={diag['ppv_lcb']:.3f} cov={diag['cov']:.3f} loser_rules={diag['loser_rules']}"
            )
            gate_summary.update({
                "tau": float(diag.get("tau", float("nan"))),
                "ppv": float(diag.get("ppv", float("nan"))),
                "ppv_lcb": float(diag.get("ppv_lcb", float("nan"))),
                "coverage": float(diag.get("cov", float("nan"))),
                "loser_rules": int(diag.get("loser_rules", 0)),
            })
        except Exception as e:
            _log(f"[{sym}] gate training failed: {e}")
    else:
        _log(f"[{sym}] gate: insufficient labeled data for training (rows={len(train_core)})")

    write_json(gate_summary_path, gate_summary)

    _write_fingerprint(
        artifacts_dir,
        "mining",
        cfg_path,
        {
            "train_months": list(train_months),
            "test_months": list(test_months),
            "cands_train_rows": int(len(cands_tr)),
            "events_train_rows": int(len(events_tr)),
            "train_samples": int(len(all_train)),
            "promoted": len(promoted),
            "rules_total": len(mining_payload.get("rules", [])),
            "tau": gate_summary.get("tau"),
            "ppv": gate_summary.get("ppv"),
            "ppv_lcb": gate_summary.get("ppv_lcb"),
            "coverage": gate_summary.get("coverage"),
            "loser_rules": gate_summary.get("loser_rules"),
        },
    )


def stage_simulate(sym: str, train_months: List[str], test_months: List[str], cfg: Dict[str, Any], cfg_path: Path, fold_dir: Path, force: bool) -> None:


    _log(f"[{sym}] simulate: start test={test_months}")


    prog_cfg = (cfg.get("progress", {}) or {}).get("sim", {}) or {}
    scheduler_fallback = False
    show_inner = bool(prog_cfg.get("inner_schedule_bar", False))
    desc_inner = str(prog_cfg.get("schedule_desc", "schedule"))
    upd_every = int(prog_cfg.get("schedule_update_every", 1000))

    sched_cfg = (cfg.get("execution_sim", {}) or {}).get("scheduler", {}) or {}
    weight_mode = str(sched_cfg.get("weight_mode", "expR"))
    sched_r_mult = float(sched_cfg.get("r_mult", 5.0))
    timeout_sec = sched_cfg.get("timeout_sec", None)
    timeout_sec = float(timeout_sec) if timeout_sec is not None else None


    spath = fold_dir / "stats.json"


    tpath = fold_dir / "trades.csv"


    trpath = fold_dir / "train_table.csv"


    if skip_if_exists(spath, force) and tpath.exists() and trpath.exists():


        _log(f"[{sym}] SKIP simulate (exists)")


        return


    # Friendly guard: require features stage to exist
    cpath = fold_dir / "candidates.parquet"
    if not cpath.exists():
        raise FileNotFoundError(
            f"{cpath} not found. Run features stage first:\n"
            f"  python -m cto10r --config {cfg_path} --mode walkforward --only features --force"
        )
    cands = pd.read_parquet(cpath)


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
    merge_cols = list(dict.fromkeys([*join_cols, *feature_cols, *literal_cols, *age_cols]))
    merge_cols = [c for c in merge_cols if c in cands_t_norm.columns]

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
    gate_tau = float("nan")
    coverage = 0.0
    empirical_ppv = float("nan")
    gate = None
    if gate_bundle.exists():
        try:
            gate = load_trained_gate(gate_bundle)
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
    # normalize p_win and boolean columns
    ev["p_win"] = pd.to_numeric(ev["p_win"], errors="coerce")
    ev["loser_mask"] = ev["loser_mask"].fillna(False).astype(bool)
    ev["enter"] = ev["enter"].fillna(False).astype(bool)
    ev["tau_used"] = ev["tau_used"].fillna(gate_tau)

    if "preempted" not in ev.columns:
        ev["preempted"] = False
    ev["preempted"] = ev["preempted"].fillna(False).astype(bool)

    ev["enter"] = ev["enter"].fillna(False) & (~ev["preempted"])

    ev = ev.sort_values("ts").reset_index(drop=True)

    decision_df = cands_t_norm.copy()
    default_enter = pd.Series(False, index=decision_df.index, dtype=bool)
    go_series = (
        pd.to_numeric(decision_df.get("enter", default_enter), errors="coerce")
        .fillna(0)
        .astype(bool)
    )

    # Robust label merge: try full key, then fallback to ts-only
    label_merge_cols = [c for c in ["ts", "side", "entry", "level", "risk_dist"] if c in decision_df.columns and c in ev.columns]

    def _merge_labels(dec_df: pd.DataFrame, ev_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            out = dec_df.copy()
            out["label_resolved"] = np.nan
            return out
        ll = ev_df[cols + ["outcome"]].copy()
        ll["label_resolved"] = ll["outcome"].map({"win": 1, "loss": -1, "timeout": 0})
        ll = ll.drop_duplicates(subset=cols, keep="last")
        out = dec_df.merge(ll[cols + ["label_resolved"]], on=cols, how="left")
        return out

    decision_df = _merge_labels(decision_df, ev, label_merge_cols)
    # Materialize label from resolved mapping
    decision_df["label"] = pd.to_numeric(decision_df.get("label", pd.Series(np.nan, index=decision_df.index)), errors="coerce")
    decision_df["label"] = decision_df["label"].fillna(decision_df.pop("label_resolved"))
    # Fallback: if nothing matched, try ts-only
    if decision_df["label"].isna().all():
        decision_df = _merge_labels(decision_df.drop(columns=["label"], errors="ignore"), ev, ["ts"])
        decision_df["label"] = decision_df.pop("label_resolved")

    decision_df["label"] = pd.to_numeric(decision_df["label"], errors="coerce")
    if "preempted" not in decision_df.columns:
        decision_df["preempted"] = False
    decision_df["preempted"] = (
        decision_df["preempted"].fillna(False).astype("boolean").astype(bool)
    )
    decision_df["enter"] = go_series.to_numpy() & (~decision_df["preempted"].to_numpy())

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
    tau_str = f"{tau_used_val:.3f}" if tau_used_val == tau_used_val else "nan"
    _log(f"[sim] tau_used={tau_str} decided_cov={coverage:.3f} decided_empirical_ppv={ppv_str}")

    ev_all = ev[~ev["preempted"]].copy()
    preempted_total = int(ev["preempted"].sum())
    total = int(len(ev_all))
    wins_all = int((ev_all["outcome"] == "win").sum())
    losses_all = int((ev_all["outcome"] == "loss").sum())
    timeouts_all = int((ev_all["outcome"] == "timeout").sum())
    denom_all = max(wins_all + losses_all, 1)
    wr_all = wins_all / denom_all if denom_all else 0.0
    expR_all = wr_all * label_r_mult + (1.0 - wr_all) * (-1.0)

    # --- PREEMPT robustness: tolerate missing column ---
    if "preempted" not in ev.columns:
        ev["preempted"] = False

    # --- Compute gross R same as before ---
    trades = ev[ev["outcome"].isin(["win", "loss"])].copy()
    trades["R_gross"] = np.where(trades["outcome"] == "win", label_r_mult, -1.0)

    # --- Fees & Funding config (live-parity) ---
    fees_cfg = (cfg.get("execution_sim", {}) or {}).get("fees", {}) or {}
    fund_cfg = (cfg.get("execution_sim", {}) or {}).get("funding", {}) or {}

    maker = float(fees_cfg.get("maker_bps", 0.0)) / 1e4
    taker = float(fees_cfg.get("taker_bps", 0.0)) / 1e4
    # Assume taker entry, maker exit by default (common on perps); allow override
    entry_is_maker = bool(fees_cfg.get("entry_is_maker", False))
    exit_is_maker  = bool(fees_cfg.get("exit_is_maker", True))

    fee_entry = maker if entry_is_maker else taker
    fee_exit  = maker if exit_is_maker  else taker
    roundtrip_fee = float(fees_cfg.get("roundtrip_bps", 0.0)) / 1e4
    if roundtrip_fee > 0.0:
        # If explicit roundtrip set, override split
        fee_entry = roundtrip_fee / 2.0
        fee_exit  = roundtrip_fee / 2.0

    # Funding settings
    fund_enabled = bool(fund_cfg.get("enabled", False))
    fund_per_hour = float(fund_cfg.get("rate_per_hour", 0.0))  # e.g. 0.0001 for 1bp/hr
    fund_sign = float(fund_cfg.get("sign", 1.0))  # +1 cost for longs by default; set -1 if your venue credits longs

    # --- Convert costs to R-units using entry and risk distance ---
    # R-unit = price move equal to risk_dist. Fee in R = (fee * entry_price) / risk_dist.
    # This cancels quantity. We guard zeros/NaNs with eps.
    eps = 1e-12
    entry = pd.to_numeric(trades.get("entry"), errors="coerce")
    risk  = pd.to_numeric(trades.get("risk_dist"), errors="coerce").abs()
    leverage = (entry / np.maximum(risk, eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fee_R = leverage * (fee_entry + fee_exit)

    # --- Funding proration by time in position ---
    if fund_enabled:
        t_start = pd.to_numeric(trades.get("ts"), errors="coerce")
        t_end   = pd.to_numeric(trades.get("outcome_ts"), errors="coerce")
        hours   = ((t_end - t_start).astype(float) / 3600.0).clip(lower=0.0)
        funding_R = leverage * fund_per_hour * hours * fund_sign
    else:
        funding_R = 0.0

    trades["R_costs"] = fee_R + (funding_R if isinstance(funding_R, (pd.Series, np.ndarray)) else float(funding_R))
    trades["R"] = trades["R_gross"] - trades["R_costs"]

    # Export with schema
    trades_export = normalize_events_schema(trades)
    trades_export.to_csv(tpath, index=False)

    cands_export = normalize_cands_schema(cands_t_norm)
    cands_export.to_csv(trpath, index=False)

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
            "preempted_events": preempted_total,
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

    # Cost-aware summary (does not change existing outputs)
    avg_cost_R = float(np.nanmean(trades["R_costs"])) if len(trades) else 0.0
    _log(
        f"[{sym}] simulate[costs]: avg_cost_R={avg_cost_R:.4f} maker={maker:.4f} taker={taker:.4f} "
        f"fund/hr={fund_per_hour:.5f} fund_on={fund_enabled}"
    )

    ev_g = ev[ev["enter"]].copy()
    gated_count = int(ev_g.shape[0])

    ev_g_export = normalize_events_schema(ev_g)
    fold_dir.mkdir(parents=True, exist_ok=True)
    trades_csv = fold_dir / "trades_gated.csv"
    presched_csv = fold_dir / "trades_gated_presched.csv"
    ev_g_export.to_csv(presched_csv, index=False)

    # Sequential scheduling: one-at-a-time, busy until exit
    selected = ev_g.sort_values(["ts", "side"]).reset_index(drop=True)
    position_open = False
    position_exit_ts = -1
    kept_rows = []
    preempted_count = 0
    for _, r in selected.iterrows():
        t = int(r.get("ts", 0))
        if position_open and t < position_exit_ts:
            preempted_count += 1
            continue
        kept_rows.append(r)
        out_ts_val = r.get("outcome_ts")
        out_ts = None
        try:
            if pd.notna(out_ts_val):
                out_ts = int(out_ts_val)
        except Exception:
            pass
        if out_ts is None:
            out_ts = t + int(float(labels_cfg_sim.get("horizon_hours", 480)) * 3600_000)
        position_open = True
        position_exit_ts = int(out_ts)
    take_sched = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else selected.iloc[0:0].copy()
    # Persist for parity tooling
    try:
        normalize_events_schema(take_sched).to_parquet(fold_dir / "sim_scheduled.parquet", index=False)
    except Exception:
        pass
    print(f"[SIM] admin_preempt(position_open)={preempted_count}")

    tw = (cfg.get("tripwires", {}) or {})
    ppv_lcb_min = float(tw.get("ppv_lcb_min", 0.50))
    timeout_rate_max = float(tw.get("timeout_rate_max", 0.60))
    window_days = int(tw.get("window_days", 3))
    fallback_days_max = int(tw.get("fallback_days_max", 3))

    tg = take_sched.copy()
    if "ts" in tg.columns:
        tscol = "ts"
        tg["_date"] = pd.to_datetime(tg[tscol], unit="s", errors="coerce").dt.date
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

    with open(fold_dir / "tripwire_status.json", "w", encoding="utf-8") as fh:
        json.dump(tripwire, fh, indent=2)

    if ppv_breach_days or to_breach_days:
        _log(f"[{sym}] TRIPWIRE breach: {tripwire}")

    take_sched_export = normalize_events_schema(take_sched)
    take_sched_export.to_csv(trades_csv, index=False)

    _log(f"[SIM] coverage={coverage:.3f} selected={gated_count} scheduled={len(take_sched)}")
    if not trades_csv.exists():
        _write_empty_trades_csv(trades_csv)
    if not presched_csv.exists():
        _write_empty_trades_csv(presched_csv)

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
            "r_mult": sched_r_mult,
            "scheduler_fallback": scheduler_fallback,
        },
    )

    _log(
        f"[{sym}] simulate[GATED]: kept={len(ev_g)} scheduled={len(take_sched)} wr={g_wr:.3f} expR={g_expR:.2f}"
        + (" [fallback]" if scheduler_fallback else "")
    )

    # --- Parity tests (hard-fail if references provided) ---
    parity = (cfg.get("tests", {}) or {}).get("parity", {}) or {}
    if bool(parity.get("enabled", False)):
        import json as _json
        import numpy as _np
        import pandas as _pd
        import sys as _sys
        r_tol = float(parity.get("r_tol", 0.02))
        max_sig_mm = int(parity.get("max_signal_mismatch", 0))
        sig_ref = parity.get("signals_ref_csv")
        trd_ref = parity.get("trades_ref_csv")

        # SAME-SIGNAL: compare enter mask if reference exists
        sig_fail = 0
        if sig_ref:
            ref = _pd.read_csv(sig_ref)
            cur = ev[["ts", "side", "enter"]].copy()
            joined = _pd.merge(cur, ref, on=["ts", "side"], how="outer", suffixes=("_cur", "_ref"))
            joined["enter_cur"] = joined["enter_cur"].fillna(False).astype(bool)
            joined["enter_ref"] = joined["enter_ref"].fillna(False).astype(bool)
            mism = (joined["enter_cur"] != joined["enter_ref"]).sum()
            sig_fail = int(mism)
            _log(f"[{sym}] parity[same-signal]: mismatches={mism}")
            if sig_fail > max_sig_mm:
                _log(f"[{sym}] parity[same-signal]=FAIL (>{max_sig_mm})")
                _sys.exit(1)
            else:
                _log(f"[{sym}] parity[same-signal]=PASS")

        # SAME-TRADE: compare exit_type and R_net if reference exists
        if trd_ref:
            ref = _pd.read_csv(trd_ref)
            cur = trades_export[["ts", "side", "outcome", "R"]].copy()
            cur = cur.rename(columns={"outcome": "exit_type", "R": "R_net"})
            ref = ref[["ts", "side", "exit_type", "R_net"]].copy()
            j = _pd.merge(cur, ref, on=["ts", "side"], how="outer", suffixes=("_cur", "_ref"))
            j["exit_type_cur"] = j["exit_type_cur"].fillna("MISSING")
            j["exit_type_ref"] = j["exit_type_ref"].fillna("MISSING")
            type_mismatch = (j["exit_type_cur"] != j["exit_type_ref"]).sum()
            r_diff = (j["R_net_cur"].fillna(_np.nan) - j["R_net_ref"].fillna(_np.nan)).abs()
            r_mismatch = int((r_diff > r_tol).sum())
            _log(f"[{sym}] parity[same-trade]: exit_type_mismatch={type_mismatch} R_mismatch(>{r_tol})={r_mismatch}")
            if type_mismatch > 0 or r_mismatch > 0:
                _log(f"[{sym}] parity[same-trade]=FAIL")
                _sys.exit(1)
            else:
                _log(f"[{sym}] parity[same-trade]=PASS")


# ------------------------ orchestrator ------------------------


def run_walkforward(cfg_path: str, force: bool, clean: bool, only: Optional[str], from_stage: Optional[str], until_stage: Optional[str]) -> None:


    cfg_path = Path(cfg_path)

    cfg = load_cfg(cfg_path)


    ensure_parquet_engine()

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


                    stage_features(sym, tr_m, te_m, cfg, cfg_path, fold_dir, force)


                except Exception as e:


                    rep = exception_to_report("features", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


            if "tick_labeling" in stages:


                try:


                    stage_tick_labeling(sym, tr_m, te_m, cfg, cfg_path, fold_dir, force)


                except Exception as e:


                    rep = exception_to_report("tick_labeling", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


            if "mining" in stages:


                try:


                    stage_mining(sym, tr_m, te_m, cfg, cfg_path, fold_dir, force)


                except Exception as e:


                    rep = exception_to_report("mining", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


            if "simulate" in stages:


                try:


                    stage_simulate(sym, tr_m, te_m, cfg, cfg_path, fold_dir, force)


                except Exception as e:


                    rep = exception_to_report("simulate", cfg, fold_dir, e)


                    (fold_dir / "crash_report.json").write_text(json.dumps(rep, indent=2))


                    raise


# ------------------------ main ------------------------


def main():


    _seed_determinism()

    args = parse_args()


    if args.mode == "preflight":


        cfg = load_cfg(args.config)


        ensure_parquet_engine()

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


