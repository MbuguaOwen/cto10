\
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from .util import coerce_epoch_ms

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
):
    # Determine win/loss/timeout using banded first-touch logic on streamed ticks
    cands = cands.sort_values("ts").reset_index(drop=True).copy()

    H_ms_base = int(horizon_hours * 3600 * 1000)
    Q_ms_base = int(quick_hours * 3600 * 1000)
    n = len(cands)

    def _with_progress(it):
        if show_progress and n > 1000:
            return tqdm(it, total=n, dynamic_ncols=True, leave=False, desc="label ticks")
        return it

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

    results = []
    all_ticks = []
    for df in ticks_iterables:
        if len(df):
            all_ticks.append(df[["ts", "price"]].sort_values("ts"))
    if not all_ticks:
        for i, row in _with_progress(cands.iterrows()):
            H_i = int(H_ms[i]) if i < len(H_ms) else H_ms_base
            results.append(
                {
                    "ts": int(row["ts"]),
                    "side": row["side"],
                    "entry": float(row["entry"]),
                    "level": float(row["level"]),
                    "risk_dist": float(row["risk_dist"]),
                    "outcome": "timeout",
                    "outcome_ts": int(row["ts"] + H_i),
                    "r1_ts": None,
                    "tp_ts": None,
                }
            )
        return pd.DataFrame(results)

    ticks = pd.concat(all_ticks, axis=0).sort_values("ts").reset_index(drop=True)
    # Build fast arrays once
    tss_all = pd.to_numeric(ticks["ts"], errors="coerce").astype("int64").to_numpy()
    px_all = pd.to_numeric(ticks["price"], errors="coerce").astype(float).to_numpy()
    busy_until = -1
    for i, row in _with_progress(cands.iterrows()):
        ts_entry = int(row["ts"])
        if non_overlap and busy_until != -1 and ts_entry < busy_until:
            results.append({
                "ts": ts_entry,
                "side": row["side"],
                "entry": float(row["entry"]),
                "level": float(row["level"]),
                "risk_dist": float(row["risk_dist"]),
                "outcome": "timeout",
                "outcome_ts": ts_entry,
                "r1_ts": None,
                "tp_ts": None,
            })
            continue
        t0 = ts_entry
        side = row["side"]
        E = float(row["entry"])
        P = float(row["level"])
        R = float(row["risk_dist"])
        tp = E + r_mult * R if side == "long" else E - r_mult * R
        r1 = E + 1.0 * R if side == "long" else E - 1.0 * R
        H_i = int(H_ms[i]) if i < len(H_ms) else H_ms_base
        Q_i = int(Q_ms[i]) if i < len(Q_ms) else Q_ms_base
        t_end = t0 + H_i
        t_r1_deadline = t0 + Q_i

        # Fast window slice via searchsorted
        lo = np.searchsorted(tss_all, t0, side="right")
        hi = np.searchsorted(tss_all, t_end, side="right")
        if hi <= lo:
            results.append({
                "ts": ts_entry,
                "side": row["side"],
                "entry": float(row["entry"]),
                "level": float(row["level"]),
                "risk_dist": float(row["risk_dist"]),
                "outcome": "timeout",
                "outcome_ts": t_end,
                "r1_ts": None,
                "tp_ts": None,
            })
            continue
        tss = tss_all[lo:hi]
        price = px_all[lo:hi]

        outcome = "timeout"
        out_ts = t_end
        r1_ts = None
        tp_ts = None

        band = stop_band_atr * float(row["atr"])
        violated = False
        viol_ix = None

        if side == "long":
            P_eff = P - band
            below = price < P_eff
            idx = np.flatnonzero(below)
            if idx.size:
                cuts = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[0, cuts + 1]
                ends = np.r_[cuts, len(idx) - 1]
                for s, e in zip(starts, ends):
                    length = e - s + 1
                    if length >= consec_ticks:
                        if (tss[idx[e]] - tss[idx[s]]) >= dwell_ms:
                            violated = True
                            viol_ix = idx[s]
                            break

            ge_r1 = np.where(price >= r1)[0]
            ge_tp = np.where(price >= tp)[0]

            if violated and viol_ix is not None and (not ge_tp.size or viol_ix < ge_tp[0]):
                outcome = "loss"
                out_ts = int(tss[viol_ix])
            else:
                if ge_r1.size and tss[ge_r1[0]] <= t_r1_deadline:
                    r1_ts = int(tss[ge_r1[0]])
                    if ge_tp.size:
                        if (not violated) or (viol_ix is not None and ge_tp[0] < viol_ix):
                            outcome = "win"
                            tp_ts = int(tss[ge_tp[0]])
                            out_ts = tp_ts
                        else:
                            outcome = "loss"
                            out_ts = int(tss[viol_ix])
                    else:
                        if violated and viol_ix is not None:
                            outcome = "loss"
                            out_ts = int(tss[viol_ix])
                        else:
                            outcome = "timeout"
                            out_ts = t_end
                else:
                    outcome = "timeout"
                    out_ts = t_end
        else:
            P_eff = P + band
            above = price > P_eff
            idx = np.flatnonzero(above)
            if idx.size:
                cuts = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[0, cuts + 1]
                ends = np.r_[cuts, len(idx) - 1]
                for s, e in zip(starts, ends):
                    length = e - s + 1
                    if length >= consec_ticks:
                        if (tss[idx[e]] - tss[idx[s]]) >= dwell_ms:
                            violated = True
                            viol_ix = idx[s]
                            break

            le_r1 = np.where(price <= r1)[0]
            le_tp = np.where(price <= tp)[0]

            if violated and viol_ix is not None and (not le_tp.size or viol_ix < le_tp[0]):
                outcome = "loss"
                out_ts = int(tss[viol_ix])
            else:
                if le_r1.size and tss[le_r1[0]] <= t_r1_deadline:
                    r1_ts = int(tss[le_r1[0]])
                    if le_tp.size:
                        if (not violated) or (viol_ix is not None and le_tp[0] < viol_ix):
                            outcome = "win"
                            tp_ts = int(tss[le_tp[0]])
                            out_ts = tp_ts
                        else:
                            outcome = "loss"
                            out_ts = int(tss[viol_ix])
                    else:
                        if violated and viol_ix is not None:
                            outcome = "loss"
                            out_ts = int(tss[viol_ix])
                        else:
                            outcome = "timeout"
                            out_ts = t_end
                else:
                    outcome = "timeout"
                    out_ts = t_end

        results.append(
            {
                "ts": int(row["ts"]),
                "side": side,
                "entry": E,
                "level": P,
                "risk_dist": R,
                "outcome": outcome,
                "outcome_ts": int(out_ts),
                "r1_ts": r1_ts,
                "tp_ts": tp_ts,
            }
        )
        if non_overlap and outcome == "win" and tp_ts is not None:
            busy_until = int(tp_ts)
    return pd.DataFrame(results)

