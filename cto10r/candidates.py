\

import numpy as np

import pandas as pd



from .util import eta_from_atr_percentile, cusum_events, rolling_sigma


AGE_COLS = [
  # feature bins
  "qbin_t_240","qbin_t_60","qbin_t_15","qbin_accel_15_240","qbin_atr","qbin_dcw",
  "qbin_body_dom","qbin_close_to_hi_atr","qbin_close_to_lo_atr","qbin_atr_p","qbin_dcw_p",
  # age of current bin
  "agebin_t_240","agebin_t_60","agebin_t_15","agebin_accel_15_240","agebin_atr","agebin_dcw",
  "agebin_body_dom","agebin_close_to_hi_atr","agebin_close_to_lo_atr","agebin_atr_p","agebin_dcw_p",
  # quantized age
  "qbin_agebin_t_240","qbin_agebin_t_60","qbin_agebin_t_15","qbin_agebin_accel_15_240",
  "qbin_agebin_atr","qbin_agebin_dcw","qbin_agebin_body_dom","qbin_agebin_close_to_hi_atr",
  "qbin_agebin_close_to_lo_atr","qbin_agebin_atr_p","qbin_agebin_dcw_p",
  # time since low/high bins
  "since_lowbin_dcw","since_highbin_dcw","since_lowbin_dcw_p","since_highbin_dcw_p",
  "since_lowbin_atr","since_highbin_atr","since_lowbin_atr_p","since_highbin_atr_p",
  # occupancy examples (include if present)
  "occ_lowbin_dcw_W60","occ_highbin_dcw_W60","occ_lowbin_dcw_W240","occ_highbin_dcw_W240",
  "occ_lowbin_dcw_p_W60","occ_highbin_dcw_p_W60","occ_lowbin_dcw_p_W240","occ_highbin_dcw_p_W240",
]


# ---------- BLIND MODE (CUSUM / STRIDE) ----------



def _thin_indices_by_stride(n: int, k: int) -> np.ndarray:

    if k <= 1:

        return np.arange(n, dtype=int)

    return np.arange(0, n, int(k), dtype=int)





def _thin_indices_by_cusum(feat: pd.DataFrame, params: dict) -> np.ndarray:

    c = feat["close"].astype(float).to_numpy()

    r = np.zeros_like(c)

    if len(c) > 1:

        safe_c = np.where(c <= 0, np.nan, c)

        r[1:] = np.diff(np.log(safe_c))

    r[~np.isfinite(r)] = 0.0



    L = int(params.get("ret_lookback", 240))

    k = float(params.get("k_sigma", 0.75))

    drift = float(params.get("drift", 0.0))

    min_gap = int(params.get("min_gap_bars", 0))

    fallback_k = int(params.get("fallback_stride_k", 0))



    sig = rolling_sigma(r, L)

    h = k * sig

    ev_ix = cusum_events(r, h, drift=drift, min_gap=min_gap)



    if fallback_k and (len(ev_ix) == 0 or len(ev_ix) < (len(c) // max(fallback_k * 10, 1))):

        stride_ix = _thin_indices_by_stride(len(c), fallback_k)

        ev_ix = np.unique(np.concatenate([ev_ix, stride_ix]))



    return ev_ix





def _safe_keep(df: pd.DataFrame, base_keep: list) -> pd.DataFrame:
    """Preserve order, append families, and de-duplicate columns.
    Families: nl_*, qbin_*, agebin_*, qbin_agebin_*, since_*, occ_*
    """
    cols: list[str] = []
    seen: set[str] = set()

    def add(name: str):
        if name in df.columns and name not in seen:
            cols.append(name)
            seen.add(name)

    # base keeps first
    for c in base_keep:
        add(c)

    # then families
    fam_prefixes = ("nl_", "qbin_", "agebin_", "qbin_agebin_", "since_", "occ_")
    for c in df.columns:
        if c.startswith(fam_prefixes):
            add(c)

    return df.loc[:, cols].copy()





def build_candidates_blind(

    feat: pd.DataFrame,

    labels_cfg: dict,

    blind_cfg: dict

) -> pd.DataFrame:

    thinning = (blind_cfg.get("thinning", {}) or {})

    ttype = str(thinning.get("type", "cusum_by_ret")).lower()

    tparams = thinning.get("params", {}) or {}

    sides = str(blind_cfg.get("sides", "both")).lower()



    labels_side = str(labels_cfg.get("side", "both")).lower()

    if labels_side in ("long", "short"):

        sides = labels_side



    base_eta = float(labels_cfg.get("eta_atr", 0.30))

    eta_tbl = labels_cfg.get("eta_by_atr_p")

    cap_cfg = labels_cfg.get("eta_cap", {})

    eta_min = float(cap_cfg.get("min", 0.0))

    eta_max = float(cap_cfg.get("max", np.inf))



    f = feat.copy()

    f["bar_idx"] = f.index

    if ttype == "stride":

        k = int(tparams.get("k", 5))

        ix = _thin_indices_by_stride(len(f), k)

    else:

        ix = _thin_indices_by_cusum(f, tparams)

    f = f.iloc[ix].copy()



    atr_p_series = f.get("atr_p", pd.Series(np.nan, index=f.index))

    eta_series = eta_from_atr_percentile(atr_p_series, base_eta, eta_tbl).astype(float)

    f["eta"] = eta_series.clip(lower=eta_min, upper=eta_max)



    out = []

    rho = float(labels_cfg.get("risk_floor_rho", 0.0))



    def _emit(side: str, g: pd.DataFrame) -> None:

        if g.empty:

            return

        x = g.copy()

        x["side"] = side

        x["entry"] = x["close"].astype(float)

        if side == "long":

            x["level"] = x["entry"] - x["eta"] * x["atr"]

        else:

            x["level"] = x["entry"] + x["eta"] * x["atr"]

        x["risk_dist"] = (x["entry"] - x["level"]).abs()

        if rho > 0:

            x = x[x["risk_dist"] >= rho * x["atr"]]

        keep = [
            "ts","entry","level","risk_dist","eta","atr","side",
            "t_240","t_60","t_15","accel_15_240","body_dom",
            "dcw","atr_p","dcw_p","close","open","high","low","ym",
            "close_to_hi_atr","close_to_lo_atr","bar_idx"
        ] + [c for c in AGE_COLS if c in g.columns]

        if not x.empty:

            out.append(_safe_keep(x, keep))



    if sides in ("both", "long"):

        _emit("long", f)

    if sides in ("both", "short"):

        _emit("short", f)



    if not out:

        return f.iloc[0:0].copy()



    cand = pd.concat(out, ignore_index=True).sort_values("ts")

    for k in ("entry", "level", "risk_dist", "eta"):

        if k in cand:

            cand[k] = cand[k].astype("float64").round(8)

    return cand.reset_index(drop=True)





def build_candidates_router(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    mode = str(cfg.get("candidates", {}).get("mode", "blind")).lower()

    labels_cfg = cfg.get("labels", {}) or {}

    if mode == "blind":

        bcfg = (cfg.get("candidates", {}).get("blind") or {})

        return build_candidates_blind(feat, labels_cfg, bcfg)



    gate_cfg = cfg.get("candidate_gate", {})

    labels_side = str(labels_cfg.get("side", "both")).lower()

    sides = ("long", "short") if labels_side == "both" else (labels_side,)

    out = []

    for side in sides:

        out.append(build_candidates(feat, labels_cfg, gate_cfg, side))

    if not out:

        return feat.iloc[0:0].copy()

    return pd.concat(out, ignore_index=True).sort_values("ts").reset_index(drop=True)





def diagnose_gates(feat: pd.DataFrame, gate_cfg: dict, side: str) -> dict:

    bd = feat['body_dom'].values >= gate_cfg['body_dom_min']

    if side == 'long':

        near_ext = feat['close_to_hi_atr'].values <= gate_cfg['close_to_ext_atr_max']

    else:

        near_ext = feat['close_to_lo_atr'].values <= gate_cfg['close_to_ext_atr_max']

    t240 = np.abs(feat['t_240'].values) >= gate_cfg['t240_abs_min']

    sign_ok = np.sign(feat['t_240'].values) == (1 if side=='long' else -1)

    accel = feat['accel_15_240'].values >= gate_cfg['accel_15_240_min']

    atr_p = feat['atr_p'].values <= gate_cfg['compression']['atr_p_max']

    dcw_p = feat['dcw_p'].values <= gate_cfg['compression']['dcw_p_max']

    base = bd & near_ext & t240 & sign_ok & accel & atr_p & dcw_p



    ks = gate_cfg['kappa_sign_thr']

    votes = np.zeros(len(feat), dtype=int)

    votes += (feat['t_240'].values >= ks['t_240_min']).astype(int) * (1 if side=='long' else 0)

    votes += (feat['t_240'].values <= -ks['t_240_min']).astype(int) * (1 if side=='short' else 0)

    votes += (feat['t_60'].values >= ks['t_60_min']).astype(int) * (1 if side=='long' else 0)

    votes += (feat['t_60'].values <= -ks['t_60_min']).astype(int) * (1 if side=='short' else 0)

    votes += (feat['t_15'].values >= ks['t_15_min']).astype(int) * (1 if side=='long' else 0)

    votes += (feat['t_15'].values <= -ks['t_15_min']).astype(int) * (1 if side=='short' else 0)

    quorum = votes >= int(ks['quorum'])



    return {

        'N': len(feat),

        'body_dom': int(bd.sum()),

        'near_ext': int(near_ext.sum()),

        't240_abs': int(t240.sum()),

        'sign_ok': int(sign_ok.sum()),

        'accel': int(accel.sum()),

        'atr_p': int(atr_p.sum()),

        'dcw_p': int(dcw_p.sum()),

        'base': int(base.sum()),

        'quorum': int(quorum.sum()),

        'final': int((base & quorum).sum()),

    }



def candidate_mask(feat: pd.DataFrame, gate_cfg: dict, side: str) -> np.ndarray:

    # shared gates

    bd = feat["body_dom"].values >= gate_cfg["body_dom_min"]

    if side == "long":

        near_ext = feat["close_to_hi_atr"].values <= gate_cfg["close_to_ext_atr_max"]

    else:

        near_ext = feat["close_to_lo_atr"].values <= gate_cfg["close_to_ext_atr_max"]

    t240 = np.abs(feat["t_240"].values) >= gate_cfg["t240_abs_min"]

    sign_ok = np.sign(feat["t_240"].values) == (1 if side=="long" else -1)

    accel = feat["accel_15_240"].values >= gate_cfg["accel_15_240_min"]

    # compression over lookback: we approximate via current percentiles columns (assumed added)

    atr_p = feat["atr_p"].values <= gate_cfg["compression"]["atr_p_max"]

    dcw_p = feat["dcw_p"].values <= gate_cfg["compression"]["dcw_p_max"]

    base = bd & near_ext & t240 & sign_ok & accel & atr_p & dcw_p

    # kappa quorum

    ks = gate_cfg["kappa_sign_thr"]

    votes = np.zeros(len(feat), dtype=int)

    votes += (feat["t_240"].values >= ks["t_240_min"]).astype(int) * (1 if side=="long" else 0)

    votes += (feat["t_240"].values <= -ks["t_240_min"]).astype(int) * (1 if side=="short" else 0)

    votes += (feat["t_60"].values >= ks["t_60_min"]).astype(int) * (1 if side=="long" else 0)

    votes += (feat["t_60"].values <= -ks["t_60_min"]).astype(int) * (1 if side=="short" else 0)

    votes += (feat["t_15"].values >= ks["t_15_min"]).astype(int) * (1 if side=="long" else 0)

    votes += (feat["t_15"].values <= -ks["t_15_min"]).astype(int) * (1 if side=="short" else 0)

    quorum = votes >= int(ks["quorum"])

    return base & quorum



def build_candidates(feat: pd.DataFrame, labels_cfg: dict, gate_cfg: dict, side: str):

    base_eta = float(labels_cfg.get("eta_atr", 0.30))

    eta_table = labels_cfg.get("eta_by_atr_p")

    eta_series = eta_from_atr_percentile(feat["atr_p"], base_eta, eta_table)



    cap_cfg = labels_cfg.get("eta_cap", {})

    eta_min = float(cap_cfg.get("min", 0.0))

    eta_max = float(cap_cfg.get("max", np.inf))

    eta_series = np.clip(eta_series.astype(float), eta_min, eta_max)



    feat_eta = feat.copy()

    feat_eta["eta"] = eta_series.values



    eta_values = feat_eta["eta"].astype(float).values

    finite_eta = np.isfinite(eta_values)

    rho = float(labels_cfg.get("risk_floor_rho", 0.0))

    if finite_eta.any():

        min_eta = float(eta_values[finite_eta].min())

        if rho > min_eta:

            print(f"[candidates] WARNING: risk_floor_rho ({rho}) > min(eta) ({min_eta:.3f}) -- this can drop all rows.", flush=True)



    mask = candidate_mask(feat_eta, gate_cfg, side)

    f = feat_eta[mask].copy()

    f["side"] = side

    f["bar_idx"] = f.index



    if side == "long":

        f["entry"] = f["close"]

        f["level"] = f["entry"] - f["eta"] * f["atr"]

    else:

        f["entry"] = f["close"]

        f["level"] = f["entry"] + f["eta"] * f["atr"]



    f["risk_dist"] = (f["entry"] - f["level"]).abs()

    f = f[f["risk_dist"] >= rho * f["atr"]]



    for k in ("entry", "level", "risk_dist", "eta"):

        if k in f:

            f[k] = f[k].astype("float64").round(8)



    keep = ["ts", "entry", "level", "risk_dist", "eta", "atr", "side", "t_240", "t_60", "t_15",

            "accel_15_240", "body_dom", "atr_p", "dcw_p", "close_to_hi_atr", "close_to_lo_atr", "bar_idx"]

    return f[keep].reset_index(drop=True)

