\

import pandas as pd

import numpy as np

import json

from pathlib import Path

from .util import (
    ema,
    true_range,
    roll_tstat,
    donchian,
    body_dom,
    close_to_ext_atr,
    percentile_rank_fit,
    coerce_epoch_ms,
    bin_numeric_by_train_quantiles,
    run_length_same,
    time_since_flag,
    rolling_occupancy,
    sgn,
    safe_div,
    zero_cross_rate,
    sign_entropy,
    rolling_high,
    rolling_low,
)
import json, os
from pathlib import Path



def load_bars_any(path: Path, schema: str = "auto") -> pd.DataFrame:

    import pandas as pd

    import numpy as np



    # Read CSV robustly (header or headerless Binance 12-col), or Parquet

    if path.suffix.lower() == ".parquet":

        df = pd.read_parquet(path)

    else:

        # peek header row

        hdr = pd.read_csv(path, nrows=0)

        cols = [c.lower() for c in hdr.columns]

        # headerless if first name looks numeric OR no obvious ts column and width >= 6

        headerless = False

        if len(cols) == 0:

            headerless = True

        else:

            first = cols[0]

            headerless = first.isdigit() or (("timestamp" not in cols and "open_time" not in cols) and len(cols) >= 6)



        if headerless:

            raw = pd.read_csv(path, header=None)

            cols12 = ["open_time","open","high","low","close","volume",

                      "close_time","quote_volume","n_trades",

                      "taker_buy_base","taker_buy_quote","ignore"]

            raw = raw.iloc[:, :min(12, raw.shape[1])]

            raw.columns = cols12[:raw.shape[1]]

            df = raw

        else:

            df = pd.read_csv(path)



    cols_map = {c.lower().strip(): c for c in df.columns}

    def _pick(*cands):
        for cand in cands:
            key = cand.lower().strip()
            if key in cols_map:
                return cols_map[key]
        return None

    ts_col = _pick("ts", "timestamp", "open_time", "time")
    open_col = _pick("open", "o", "open_price")
    high_col = _pick("high", "h", "high_price")
    low_col = _pick("low", "l", "low_price")
    close_col = _pick("close", "c", "close_price")

    required = [ts_col, open_col, high_col, low_col, close_col]
    if any(col is None for col in required):
        raise ValueError(f"Bars file missing required columns ts/open/high/low/close (got: {list(df.columns)})")

    df = df.rename(
        columns={
            ts_col: "ts",
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
        }
    )

    df["ts"] = coerce_epoch_ms(df["ts"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    df["ts"] = df["ts"].astype("int64")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float64")

    out = df[["ts", "open", "high", "low", "close"]].sort_values("ts").reset_index(drop=True)

    return out



def build_features(df: pd.DataFrame, atr_n: int, donch_candidate: int, horizons_min):

    df = df.sort_values("ts").reset_index(drop=True).copy()

    o,h,l,c = [df[k].values.astype(float) for k in ["open","high","low","close"]]

    c_prev = np.r_[np.nan, c[:-1]]

    tr = true_range(h,l,c,c_prev)

    atr = ema(tr, atr_n)

    ret = np.r_[np.nan, np.diff(np.log(c))]

    t240 = roll_tstat(ret, 240)

    t60 = roll_tstat(ret, 60)

    t15 = roll_tstat(ret, 15)

    accel_15_240 = t15 - t240

    dch, dcl, dcw = donchian(h,l,donch_candidate)

    feat = df.copy()

    feat["atr"] = atr
    feat["true_range"] = tr

    # keep legacy 'ret' and provide explicit 1m log return
    feat["ret"] = ret
    feat["ret_1m"] = ret

    feat["t_240"] = t240

    feat["t_60"] = t60

    feat["t_15"] = t15

    feat["accel_15_240"] = accel_15_240

    feat["dch"] = dch

    feat["dcl"] = dcl

    feat["dcw"] = dcw

    feat["body_dom"] = body_dom(o,c,atr)

    feat["close_to_hi_atr"] = (h - c) / np.where(atr==0, np.nan, atr)

    feat["close_to_lo_atr"] = (c - l) / np.where(atr==0, np.nan, atr)

    return feat


def _batch_concat(base: pd.DataFrame, cols: dict) -> pd.DataFrame:
    return pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1, copy=False)


def add_nonlinear_features(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    nl_cfg = (cfg.get("features", {}).get("nonlinear", {}) or {})
    if not nl_cfg.get("enabled", False):
        return feat
    W_small = int(nl_cfg.get("windows", {}).get("small", 15))
    W_mid   = int(nl_cfg.get("windows", {}).get("mid", 60))
    include = list(nl_cfg.get("include", []))

    f = feat.copy()
    new = {}

    have = lambda c: (c in f.columns)

    # A) Trend shape
    if "convexity_15_60_240" in include and all(have(c) for c in ["t_15","t_60","t_240"]):
        new["nl_convexity_15_60_240"] = f["t_15"] - 2.0*f["t_60"] + f["t_240"]
    if "trend_agree_15_60" in include and all(have(c) for c in ["t_15","t_60"]):
        new["nl_trend_agree_15_60"] = (sgn(f["t_15"]) == sgn(f["t_60"])).astype("int8")
    if "trend_agree_60_240" in include and all(have(c) for c in ["t_60","t_240"]):
        new["nl_trend_agree_60_240"] = (sgn(f["t_60"]) == sgn(f["t_240"])).astype("int8")
    if "trend_ratio_15_60" in include and all(have(c) for c in ["t_15","t_60"]):
        new["nl_trend_ratio_15_60"] = safe_div(f["t_15"].abs(), f["t_60"].abs())

    # B) Compression→release
    if "squeeze_score" in include and have("dcw_p"):
        new["nl_squeeze_score"] = 100.0 - f["dcw_p"]
    if "squeeze_pressure" in include and all(have(c) for c in ["dcw_p","agebin_dcw_p"]):
        # if age already exists, finalize pressure now; otherwise compute later
        pass
    if "release_slope_15" in include and have("dcw_p"):
        new["nl_release_slope_15"] = f["dcw_p"].diff(W_small)

    # C) Location×impulse
    if "pin_high" in include and have("close_to_hi_atr"):
        new["nl_pin_high"] = (f["close_to_hi_atr"] <= 0.25).astype("int8")
    if "pin_low" in include and have("close_to_lo_atr"):
        new["nl_pin_low"]  = (f["close_to_lo_atr"] <= 0.25).astype("int8")
    if "impulse_near_high" in include and all(have(c) for c in ["body_dom","close_to_hi_atr"]):
        new["nl_impulse_near_high"] = f["body_dom"] * (f["close_to_hi_atr"] <= 0.25).astype("int8")
    if "impulse_near_low" in include and all(have(c) for c in ["body_dom","close_to_lo_atr"]):
        new["nl_impulse_near_low"]  = f["body_dom"] * (f["close_to_lo_atr"] <= 0.25).astype("int8")
    if "pos_01" in include and all(have(c) for c in ["close_to_hi_atr","close_to_lo_atr"]):
        new["nl_pos_01"] = safe_div(f["close_to_hi_atr"], (f["close_to_hi_atr"] + f["close_to_lo_atr"]))

    # D) Choppiness
    if "zcr_60" in include and have("ret_1m"):
        new["nl_zcr_60"] = zero_cross_rate(f["ret_1m"], W_mid)
    if "sign_entropy_60" in include and have("ret_1m"):
        new["nl_sign_entropy_60"] = sign_entropy(f["ret_1m"], W_mid)
    if "dc_count_60" in include and have("ret_1m"):
        s = pd.to_numeric(f["ret_1m"], errors="coerce").fillna(0.0)
        flips = (np.sign(s) != np.sign(s.shift(1))).astype("int8")
        new["nl_dc_count_60"] = flips.rolling(W_mid, min_periods=1).sum()

    # E) Vol structure
    if "rv60_over_atr" in include and all(have(c) for c in ["ret_1m","atr"]):
        rv60 = pd.to_numeric(f["ret_1m"], errors="coerce").rolling(W_mid, min_periods=10).std()
        new["nl_rv60_over_atr"] = safe_div(rv60, f["atr"])
    if "range_ratio_60" in include and have("atr") and all(have(c) for c in ["high","low"]):
        hh = rolling_high(f["high"], W_mid); ll = rolling_low(f["low"], W_mid)
        new["nl_range_ratio_60"] = safe_div(hh - ll, f["atr"])
    if "volofvol_proxy" in include and have("true_range"):
        tr5 = pd.to_numeric(f["true_range"], errors="coerce").rolling(5, min_periods=3).mean()
        new["nl_volofvol_proxy"] = tr5.rolling(12, min_periods=6).std()

    if new:
        f = _batch_concat(f, new)

    # finalize squeeze_pressure if requested and dcw_p age exists
    if "squeeze_pressure" in include and all(c in f.columns for c in ["nl_squeeze_score","agebin_dcw_p"]):
        f["nl_squeeze_pressure"] = f["nl_squeeze_score"] * f["agebin_dcw_p"].astype("float32")

    return f



def fit_percentiles(train_feat: pd.DataFrame):

    f_atr = percentile_rank_fit(train_feat["atr"].values)

    f_dcw = percentile_rank_fit(train_feat["dcw"].values)

    return f_atr, f_dcw



def apply_percentiles(feat: pd.DataFrame, f_atr, f_dcw):

    out = feat.copy()

    out["atr_p"] = f_atr(out["atr"].values)

    out["dcw_p"] = f_dcw(out["dcw"].values)

    return out



def add_age_features(feat: pd.DataFrame, cfg: dict, train_months: list, fold_dir: Path) -> pd.DataFrame:
    age_cfg = (cfg.get("features", {}).get("age", {}) or {})
    if not age_cfg.get("enabled", False):
        return feat

    # Collect base features from cfg.features.base_features (semicolon-separated lines)
    base_feats_cfg = cfg.get("features", {}).get("base_features", [])
    base_feats: list[str] = []
    for line in base_feats_cfg:
        base_feats.extend([c.strip() for c in str(line).split(";") if c.strip()])

    # Optionally include all computed non-linear features
    nl_age = bool(cfg.get("features", {}).get("nonlinear", {}).get("age_for_nonlinear", True))
    if nl_age:
        base_feats += [c for c in feat.columns if c.startswith("nl_")]

    # de-dup while preserving order
    seen = set()
    base_feats = [x for x in base_feats if not (x in seen or seen.add(x))]

    nbf       = int(age_cfg.get("n_bins_feature", 5))
    nab       = int(age_cfg.get("n_bins_age", 6))
    occ_ws    = list(age_cfg.get("occ_windows", []))
    log_age   = bool(age_cfg.get("log_age_bins", False))
    save_schema = bool(age_cfg.get("save_schema", True))
    schema_rel  = str(age_cfg.get("schema_file", "artifacts/quantiles.json"))
    schema_path = fold_dir / schema_rel

    out = feat.copy()
    is_tr = out["ym"].isin(train_months) if "ym" in out.columns else pd.Series(True, index=out.index)
    mask_tr = is_tr.to_numpy() if len(out) else np.array([], dtype=bool)

    schema = {"feature_bins": {}, "age_bins": {}}
    new_cols: dict[str, np.ndarray | pd.Series] = {}

    for col in base_feats:
        if col not in out.columns:
            continue

        # 1) Feature quantile bins (fit on TRAIN, apply to all)
        bins, edges = bin_numeric_by_train_quantiles(out.loc[is_tr, col], out[col], nbf)
        new_cols[f"qbin_{col}"] = bins.astype(np.int16)
        schema["feature_bins"][col] = edges.tolist()

        # 2) Age of current bin (run-length of current bin value)
        age = run_length_same(bins)
        new_cols[f"agebin_{col}"] = age.astype(np.int32)

        # 3) Time-since extremes (low/high bins from TRAIN-driven edges)
        low_flag  = (bins == 0)
        high_flag = (bins == nbf - 1)
        new_cols[f"since_lowbin_{col}"]  = time_since_flag(low_flag).astype("float32")
        new_cols[f"since_highbin_{col}"] = time_since_flag(high_flag).astype("float32")

        # 4) Rolling occupancy of low/high bins over windows
        for W in occ_ws:
            new_cols[f"occ_lowbin_{col}_W{W}"]  = rolling_occupancy(low_flag,  int(W)).astype("float32")
            new_cols[f"occ_highbin_{col}_W{W}"] = rolling_occupancy(high_flag, int(W)).astype("float32")

        # 5) Quantize age itself using TRAIN ages (equal-frequency bins)
        train_age = pd.Series(age[mask_tr]) if len(age) else pd.Series(dtype=float)
        all_age   = np.log1p(age) if log_age else age
        abins, aedges = bin_numeric_by_train_quantiles(train_age, pd.Series(all_age), nab)
        new_cols[f"qbin_agebin_{col}"] = abins.astype(np.int16)
        schema["age_bins"][f"agebin_{col}"] = aedges.tolist()

    if new_cols:
        out = _batch_concat(out, new_cols)

    # If squeeze_pressure requested, finalize after age columns are present
    if "nl_squeeze_score" in out.columns and "agebin_dcw_p" in out.columns and "nl_squeeze_pressure" not in out.columns:
        out["nl_squeeze_pressure"] = out["nl_squeeze_score"] * out["agebin_dcw_p"].astype("float32")

    # Age-bin clamp and causal squeeze-pressure handling
    if "agebin" in out.columns:
        out["agebin"] = out["agebin"].fillna(0).astype("int64")
        out.loc[out["agebin"] < 1, "agebin"] = 1
    if "nl_squeeze_pressure" in out.columns and "agebin" in out.columns:
        # keep causality, no leakage from NaNs
        out["nl_squeeze_pressure"] = out["nl_squeeze_pressure"].fillna(0)

    if save_schema and schema["feature_bins"]:
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

    return out
