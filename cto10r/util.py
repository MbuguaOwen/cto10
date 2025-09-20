\
import numpy as np
import pandas as pd

import math
import traceback, sys, json, os, time
from dataclasses import dataclass
from typing import Tuple

def eta_from_atr_percentile(atr_p: pd.Series, default_eta: float, table: list | None) -> pd.Series:
    '''Piecewise mapping from ATR percentile to eta (SL = eta * ATR).
    table: list of [lo_inclusive, hi_exclusive, eta_value] in percent (0..100].
    If table is None/empty, return default_eta broadcast.
    '''
    if table is None or len(table) == 0:
        idx = atr_p.index if hasattr(atr_p, 'index') else None
        return pd.Series(default_eta, index=idx, dtype=float)
    eta = pd.Series(default_eta, index=atr_p.index, dtype=float)
    ap = atr_p.astype(float).fillna(50.0)
    for lo, hi, val in table:
        mask = (ap >= float(lo)) & (ap < float(hi))
        eta.loc[mask] = float(val)
    return eta


def ema(arr, n):
    """
    NaN-tolerant EMA:
    - Seed from the first non-NaN value.
    - If input x is NaN at step i, carry previous EMA forward (no resets).
    """
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[:] = np.nan
    if len(arr) == 0:
        return out
    idx = np.flatnonzero(~np.isnan(arr))
    if idx.size == 0:
        return out
    s = float(arr[idx[0]])
    out[idx[0]] = s
    for i in range(idx[0] + 1, len(arr)):
        x = arr[i]
        s = alpha * (s if np.isnan(x) else float(x)) + (1.0 - alpha) * s
        out[i] = s
    return out

def true_range(h, l, c, c_prev):
    """
    Vectorized True Range:
      TR = max( H - L, |H - C_prev|, |L - C_prev| )
    Seed the very first element with H-L to avoid NaN cascades.
    """
    tr1 = h - l
    tr2 = np.abs(h - c_prev)
    tr3 = np.abs(l - c_prev)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    if len(tr):
        tr[0] = tr1[0]
    return tr

def roll_tstat(x, n):
    s = pd.Series(x)
    m = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=1)
    t = (m * np.sqrt(max(n,1))) / sd
    return t.values

def donchian(h, l, n):
    hh = pd.Series(h).rolling(n).max().values
    ll = pd.Series(l).rolling(n).min().values
    return hh, ll, (hh - ll)

def safe_write(path, writer_fn):
    tmp = str(path) + ".tmp"
    writer_fn(tmp)
    os.replace(tmp, str(path))

def exception_to_report(step, cfg_dict, out_dir, e):
    rep = {
        "step": step,
        "exception": str(type(e).__name__),
        "message": str(e),
        "traceback": traceback.format_exc(),
        "config_snapshot": cfg_dict,
        "env": {
            "python": sys.version,
        },
        "written_files": []
    }
    try:
        # enumerate existing files
        w = []
        for base, _, files in os.walk(out_dir):
            for f in files:
                w.append(str(os.path.join(base, f)))
        rep["written_files"] = w
    except Exception:
        pass
    return rep

def body_dom(o, c, atr):
    atr = np.where(atr==0, np.nan, atr)
    return np.abs(c - o) / atr

def close_to_ext_atr(long_side, h, l, c, atr):
    atr = np.where(atr==0, np.nan, atr)
    if long_side:
        return (h - c) / atr
    else:
        return (c - l) / atr


def coerce_epoch_ms(ts_like: pd.Series) -> pd.Series:
    """
    Return epoch timestamps in milliseconds as float64, tolerating NaNs.
    Handles inputs in seconds / milliseconds / microseconds / nanoseconds.
    - If all values are NaN, returns an all-NaN float series (same shape).
    - Does NOT cast to int; callers can dropna/cast if needed.
    """
    s = pd.to_numeric(ts_like, errors="coerce").astype("float64")
    if len(s) == 0:
        return s  # empty

    vals = s.values
    finite = np.isfinite(vals)
    if not finite.any():
        return s  # all NaN -> leave as float NaNs

    maxv = float(np.nanmax(vals[finite]))

    if maxv < 1e12:               # seconds -> ms
        s.loc[finite] = s.loc[finite] * 1000.0
    elif maxv >= 1e17:            # nanoseconds -> ms
        s.loc[finite] = s.loc[finite] / 1_000_000.0
    elif maxv >= 1e14:            # microseconds -> ms
        s.loc[finite] = s.loc[finite] / 1000.0

    return s  # float64 with NaNs preserved

def infer_bar_seconds(ts_any: pd.Series) -> int:
    """Infer bar size in seconds from any epoch unit (s/ms/microseconds/ns)."""
    ts_ms = coerce_epoch_ms(ts_any).dropna().astype("int64")
    if len(ts_ms) < 2:
        return 60
    diffs_ms = pd.Series(ts_ms).diff().dropna()
    if not len(diffs_ms):
        return 60
    # mode-ish diff in ms, fallback median
    mode = diffs_ms.mode()
    val_ms = int(mode.iloc[0]) if len(mode) else int(diffs_ms.median())
    return max(1, round(val_ms / 1000))


def percentile_rank_fit(x):
    vals = np.asarray(x[~np.isnan(x)])
    if len(vals) == 0:
        return lambda v: np.nan
    vals = np.sort(vals)
    def f(v):
        i = np.searchsorted(vals, v, side="right")
        return (i / len(vals)) * 100.0
    return np.vectorize(f, otypes=[float])


def cusum_events(ret: np.ndarray, h: np.ndarray, drift: float = 0.0, min_gap: int = 0) -> np.ndarray:
    """
    Symmetric CUSUM with possibly time-varying threshold h[t] (>=0).
    Returns indices (int) where a positive or negative excursion is detected.
    min_gap: drop events that are within <min_gap> bars of the previous event.
    """
    n = len(ret)
    if n == 0:
        return np.zeros(0, dtype=int)
    h = np.asarray(h, dtype=float)
    if h.shape[0] != n:
        raise ValueError("h must be same length as ret")

    s_pos = 0.0
    s_neg = 0.0
    events = []
    last_ix = -10**9

    for t in range(n):
        x = float(ret[t])
        thr = float(h[t])
        # positive leg
        s_pos = max(0.0, s_pos + x - drift)
        # negative leg
        s_neg = min(0.0, s_neg + x + drift)

        hit = False
        if s_pos > thr:
            hit = True
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -thr:
            hit = True
            s_pos = 0.0
            s_neg = 0.0

        if hit:
            if (t - last_ix) >= int(min_gap):
                events.append(t)
                last_ix = t

    return np.array(events, dtype=int)


def rolling_sigma(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-safe rolling std (population) with forward/backward fill at edges."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, 0.0, dtype=float)
    if win <= 1 or n == 0:
        return out
    s = pd.Series(x)
    out = s.rolling(win, min_periods=max(2, win // 3)).std(ddof=0).to_numpy()
    if np.isnan(out[0]):
        first = np.nanmin(out)
        if np.isnan(first):
            first = np.nanstd(x)
        out = np.where(np.isnan(out), first, out)
    out = np.where(~np.isfinite(out), np.nanstd(x), out)
    out = np.maximum(out, 1e-12)
    return out

def fit_quantile_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(s) == 0 or n_bins < 2:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(s, qs, method="linear")
    edges[0] = -np.inf
    edges[-1] = np.inf
    # enforce strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-12
    return edges.astype(float)

def apply_qbins(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy()
    return np.clip(np.digitize(x, edges, right=False) - 1, 0, len(edges)-2).astype(np.int16)

def run_length_same(arr: np.ndarray) -> np.ndarray:
    n = len(arr); out = np.zeros(n, dtype=np.int32); prev = None
    for i in range(n):
        v = arr[i]
        out[i] = 1 if (i == 0 or v != prev) else (out[i-1] + 1)
        prev = v
    return out

def time_since_flag(flag: np.ndarray) -> np.ndarray:
    n = len(flag); out = np.full(n, np.nan, dtype=float); last = -1
    for i in range(n):
        if flag[i]: last = i
        out[i] = (i - last) if last >= 0 else np.nan
    return out

def rolling_occupancy(flag: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(flag.astype(np.int8))
    return s.rolling(window, min_periods=1).mean().to_numpy()

def bin_numeric_by_train_quantiles(train_vals: pd.Series, all_vals: pd.Series, n_bins: int):
    edges = fit_quantile_edges(train_vals, n_bins)
    bins  = apply_qbins(all_vals, edges)
    return bins, edges

# --- non-linear helpers ---
def sgn(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    return (v > 0).astype(np.int8) - (v < 0).astype(np.int8)

def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return pd.to_numeric(a, errors="coerce") / (pd.to_numeric(b, errors="coerce").abs() + eps)

def zero_cross_rate(x: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").fillna(0.0)
    sign = s.where(s==0, np.sign(s))
    flips = (sign * sign.shift(1)).lt(0).astype("int8")
    return flips.rolling(window, min_periods=1).mean()

def sign_entropy(x: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").fillna(0.0)
    up = (s > 0).astype("int8")
    p = up.rolling(window, min_periods=1).mean().clip(1e-6, 1-1e-6)
    H = -(p*np.log(p) + (1-p)*np.log(1-p))
    H_max = - (0.5*np.log(0.5) + 0.5*np.log(0.5))
    return H / H_max

def rolling_high(x: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").rolling(n, min_periods=1).max()

def rolling_low(x: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").rolling(n, min_periods=1).min()
