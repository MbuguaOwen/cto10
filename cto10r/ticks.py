from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional

from .util import coerce_epoch_ms


def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def iter_ticks_files(ticks_dir: Path, symbol: str, yms: list, patt: str):
    """Yield (ym, path) for existing tick files, auto-detecting CSV/Parquet.
    If the given pattern doesn't exist, try swapping .csv/.parquet.
    """
    for ym in yms:
        yyyy, mm = ym.split("-")
        rel = patt.format(SYMBOL=symbol, YYYY=yyyy, MM=mm)
        path = ticks_dir / rel
        if path.exists():
            yield ym, path
            continue
        # try extension flip
        if path.suffix.lower() == ".csv":
            alt = path.with_suffix(".parquet")
            if alt.exists():
                yield ym, alt
                continue
        elif path.suffix.lower() == ".parquet":
            alt = path.with_suffix(".csv")
            if alt.exists():
                yield ym, alt


def stream_ticks_window(path: Path, tmin_ms: int, tmax_ms: int, chunk_rows: int = 2_000_000):
    # ---------- Parquet ----------
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        cols = [c.lower() for c in df.columns]
        if "ts" not in cols and "timestamp" in cols:
            df = df.rename(columns={df.columns[cols.index("timestamp")]: "ts"})
        if "price" not in cols and "last_price" in cols:
            df = df.rename(columns={df.columns[cols.index("last_price")]: "price"})
        if "ts" not in df.columns or "price" not in df.columns:
            return

        ts = coerce_epoch_ms(df["ts"])
        finite = np.isfinite(ts.values)
        if not finite.any():
            return
        df = df.loc[finite].copy()
        df["ts"] = ts.loc[finite]

        df = df[(df["ts"] >= tmin_ms) & (df["ts"] <= tmax_ms)]
        if len(df):
            yield df[["ts", "price"]]
        return

    # ---------- CSV (chunked) ----------
    for chunk in pd.read_csv(path, chunksize=chunk_rows):
        cols = [c.lower() for c in chunk.columns]
        if "ts" not in cols and "timestamp" in cols:
            chunk = chunk.rename(columns={chunk.columns[cols.index("timestamp")]: "ts"})
            cols = [c.lower() for c in chunk.columns]
        if "price" not in cols and "last_price" in cols:
            chunk = chunk.rename(columns={chunk.columns[cols.index("last_price")]: "price"})
            cols = [c.lower() for c in chunk.columns]
        if "ts" not in chunk.columns or "price" not in chunk.columns:
            continue

        ts = coerce_epoch_ms(chunk["ts"])
        finite = np.isfinite(ts.values)
        if not finite.any():
            continue

        chunk = chunk.loc[finite].copy()
        chunk["ts"] = ts.loc[finite]

        sel = chunk[(chunk["ts"] >= tmin_ms) & (chunk["ts"] <= tmax_ms)][["ts", "price"]]
        if len(sel):
            yield sel



def label_events_from_ticks(
    cands: pd.DataFrame,
    ticks_iterables,
    bar_seconds: int,
    horizon_hours: int,
    quick_hours: int,
    non_overlap: bool,
    stop_band_atr: float = 0.0,
    consec_ticks: int = 1,
    dwell_ms: int = 0,
    base_eta_for_time_scale: float | None = None,
    time_scale_clamp: tuple[float, float] | None = None,
    r_mult: float = 10.0,
    show_progress: bool = True,
    policy: str = "quick",
    intra_eps_ms: int = 1,
    min_risk_frac: float = 0.001,
    no_ticks_policy: str = "timeout",
):
    """Label candidate events using streamed ticks with robust bookkeeping."""
    cands = cands.sort_values("ts").reset_index(drop=True).copy()

    H_ms_base = int(horizon_hours * 3600 * 1000)
    Q_ms_base = int(quick_hours * 3600 * 1000)
    n = len(cands)

    def _with_progress(it):
        if show_progress and n > 1000:
            return tqdm(it, total=n, dynamic_ncols=True, leave=False, desc="label ticks")
        return it

    def _resolve_params(row):
        side = str(row.get("side", ""))
        entry = _safe_float(row.get("entry"), default=np.nan)
        level = _safe_float(row.get("level"), default=np.nan)
        if not np.isfinite(entry):
            entry = 0.0
        if not np.isfinite(level):
            level = entry
        atr_val = _safe_float(row.get("atr"), default=np.nan)
        risk = _safe_float(row.get("risk_dist"), default=np.nan)
        if not np.isfinite(risk) or risk <= 0:
            eta_val = _safe_float(row.get("eta"), default=np.nan)
            if np.isfinite(eta_val) and eta_val > 0 and np.isfinite(atr_val) and atr_val > 0:
                risk = eta_val * atr_val
            elif level != entry:
                risk = abs(entry - level)
            else:
                risk = 1e-6
        return side, float(entry), float(level), float(risk), float(atr_val) if np.isfinite(atr_val) else np.nan

    if base_eta_for_time_scale is not None and "eta" in cands.columns:
        base_eta_val = float(base_eta_for_time_scale) if float(base_eta_for_time_scale) != 0.0 else 1.0
        scale = cands["eta"].astype(float).to_numpy() / base_eta_val
        if time_scale_clamp is not None:
            lo, hi = float(time_scale_clamp[0]), float(time_scale_clamp[1])
            scale = np.clip(scale, lo, hi)
        H_ms = np.maximum(1, np.round(H_ms_base * scale)).astype("int64")
        Q_ms = np.maximum(1, np.round(Q_ms_base * scale)).astype("int64")
    else:
        H_ms = np.full(n, H_ms_base, dtype="int64")
        Q_ms = np.full(n, Q_ms_base, dtype="int64")

    results: list[dict[str, object]] = []
    no_ticks_windows = 0
    fallback_risk_count = 0

    all_ticks: list[pd.DataFrame] = []
    for df in ticks_iterables:
        if len(df):
            all_ticks.append(df[["ts", "price"]].sort_values("ts"))

    if not all_ticks:
        for i, row in _with_progress(cands.iterrows()):
            side, entry, level, risk, _ = _resolve_params(row)
            H_i = int(H_ms[i]) if i < len(H_ms) else H_ms_base
            results.append({
                "ts": int(row["ts"]),
                "side": side,
                "entry": entry,
                "level": level,
                "risk_dist": risk,
                "outcome": "timeout",
                "outcome_ts": int(int(row["ts"]) + H_i),
                "r1_ts": None,
                "tp_ts": None,
                "preempted": False,
            })
        no_ticks_windows = len(results)
        df = pd.DataFrame(results)
        df.attrs["no_ticks_windows"] = no_ticks_windows
        if "preempted" not in df.columns:
            df["preempted"] = False
        df["preempted"] = df["preempted"].fillna(False).astype(bool)
        print(f"[ticks] diagnostics: no_ticks_windows={no_ticks_windows}")
        return df

    ticks = pd.concat(all_ticks, axis=0).sort_values("ts").reset_index(drop=True)
    tss_all = pd.to_numeric(ticks["ts"], errors="coerce").astype("int64").to_numpy()
    px_all = pd.to_numeric(ticks["price"], errors="coerce").astype(float).to_numpy()

    # --- BEGIN: risk + first-touch resolution (robust & side-correct) ---
    tp_any = 0
    sl_any = 0
    tp_first = 0
    sl_first = 0
    no_move = 0
    busy_until = -1
    for i, row in _with_progress(cands.iterrows()):
        ts_entry = int(row["ts"])
        side, entry, level, risk, atr_val = _resolve_params(row)
        # Risk fallback with ATR-based floor (count diagnostics) and stop band
        R = _safe_float(row.get("risk_dist"), default=np.nan)
        if not np.isfinite(R) or R <= 0:
            atr = _safe_float(row.get("atr"), default=np.nan)
            eta = _safe_float(row.get("eta"), default=np.nan)
            if np.isfinite(atr) and atr > 0 and np.isfinite(eta) and eta > 0:
                R = eta * atr
            elif np.isfinite(atr) and atr > 0:
                R = atr
            else:
                R = _safe_float(abs(_safe_float(row.get("entry")) - _safe_float(row.get("level"))), default=np.nan)
            fallback_risk_count += 1
        # enforce ATR-based floor if ATR present
        atr_for_floor = _safe_float(row.get("atr"), default=np.nan)
        if np.isfinite(atr_for_floor) and atr_for_floor > 0:
            R = max(R, float(min_risk_frac) * atr_for_floor)
            R = max(R, float(stop_band_atr) * atr_for_floor)
        if not np.isfinite(R) or R <= 0:
            R = 1e-6
        risk = float(R)
        atr_use = atr_val if np.isfinite(atr_val) else 0.0

        # Overlaps allowed: do not administratively preempt due to busy windows.

        tp = entry + r_mult * risk if side == "long" else entry - r_mult * risk
        sl = entry - 1.0 * risk if side == "long" else entry + 1.0 * risk

        H_i = int(H_ms[i]) if i < len(H_ms) else H_ms_base
        Q_i = int(Q_ms[i]) if i < len(Q_ms) else Q_ms_base
        t_end = ts_entry + H_i
        t_r1_deadline = ts_entry + Q_i

        # Shift window start by +ε to forbid intra-bar peeking at entry tick
        slice_start = int(ts_entry) + int(intra_eps_ms)
        # find lo index strictly after entry
        lo = np.searchsorted(tss_all, slice_start, side="left")
        hi = np.searchsorted(tss_all, t_end, side="right")
        if hi <= lo:
            no_ticks_windows += 1
            base_row = {
                "ts": ts_entry,
                "side": side,
                "entry": entry,
                "level": level,
                "risk_dist": risk,
                "r1_ts": None,
                "tp_ts": None,
            }
            # admin preempt only for no-ticks windows when policy=='skip'
            is_admin = (str(no_ticks_policy).lower() == "skip")
            results.append({
                **base_row,
                "outcome": "timeout",
                "preempted": bool(is_admin),
                "outcome_ts": int(t_end),
            })
            continue

        tss = tss_all[lo:hi]
        price = px_all[lo:hi]

        # crossings (first-touch)
        if side == "long":
            idx_tp = np.where(price >= tp)[0]
            idx_sl = np.where(price <= sl)[0]
        else:
            idx_tp = np.where(price <= tp)[0]
            idx_sl = np.where(price >= sl)[0]

        has_tp = idx_tp.size > 0
        has_sl = idx_sl.size > 0
        tp_any += int(has_tp)
        sl_any += int(has_sl)

        outcome = "timeout"; out_ts = t_end; r1_ts = None; tp_ts = None
        if (not has_tp) and (not has_sl):
            no_move += 1
        elif has_tp and (not has_sl):
            tp_first += 1
            tp_ts = int(tss[int(idx_tp[0])]); out_ts = tp_ts; outcome = "win"
        elif has_sl and (not has_tp):
            sl_first += 1
            out_ts = int(tss[int(idx_sl[0])]); outcome = "loss"
        else:
            if int(idx_tp[0]) <= int(idx_sl[0]):
                tp_first += 1
                tp_ts = int(tss[int(idx_tp[0])]); out_ts = tp_ts; outcome = "win"
            else:
                sl_first += 1
                out_ts = int(tss[int(idx_sl[0])]); outcome = "loss"

        results.append({
            "ts": ts_entry,
            "side": side,
            "entry": entry,
            "level": level,
            "risk_dist": risk,
            "outcome": outcome,
            "outcome_ts": int(out_ts),
            "r1_ts": None,
            "tp_ts": tp_ts,
            "preempted": False,
        })

        if non_overlap:
            pol = str(policy).lower()
            if pol == "quick":
                busy_until = max(busy_until, int(t_r1_deadline))
                if outcome == "win" and tp_ts is not None and np.isfinite(tp_ts):
                    busy_until = max(busy_until, int(tp_ts))
            elif pol == "to_exit":
                busy_until = max(busy_until, int(out_ts))
            elif pol == "none":
                pass

    print(f"[ticks/sanity] tp_any={tp_any}/{len(cands)} sl_any={sl_any}/{len(cands)} tp_first={tp_first} sl_first={sl_first} no_move={no_move} fallback_risk={fallback_risk_count} no_ticks_windows={no_ticks_windows}")
    # --- END: risk + first-touch resolution (robust & side-correct) ---
    df = pd.DataFrame(results)
    df.attrs["no_ticks_windows"] = no_ticks_windows
    df.attrs["fallback_risk_count"] = int(fallback_risk_count)
    if "preempted" not in df.columns:
        df["preempted"] = False
    df["preempted"] = df["preempted"].fillna(False).astype(bool)
    print(f"[ticks] diagnostics: no_ticks_windows={no_ticks_windows} fallback_risk={fallback_risk_count}")
    return df
