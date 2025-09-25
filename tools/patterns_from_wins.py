# tools/patterns_from_wins.py
import os, math, argparse, json
import numpy as np
import pandas as pd
from itertools import combinations

COLUMNS = ["feature","pattern","support","wins","wr","lcb","lift","delta_logit"]

def wilson_lcb(k:int, n:int, z:float=1.96)->float:
    if n <= 0: return float("nan")
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = (z * ((p*(1-p)/n + (z*z)/(4*n*n))**0.5)) / denom
    return max(0.0, center - margin)

def logit(p: float, eps: float=1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))

def safe_df(rows, cols=COLUMNS):
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).fillna(np.nan)

def detect_binary_series(s: pd.Series, name:str) -> bool:
    if s.dtype == bool:
        return True
    vals = pd.Series(s).dropna().unique()
    try:
        vals = set(np.unique(vals))
    except Exception:
        return False
    if len(vals) <= 10 and vals.issubset({0,1}):
        return True
    if str(name).startswith(("B_","qbin_","agebin_")):
        return True
    return False

def quantile_edges(x: pd.Series, q:int=10):
    s = pd.to_numeric(x, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    if s.nunique() < 2:
        return None
    qs = np.linspace(0,1, min(q, max(2, s.nunique())))
    edges = np.unique(np.quantile(s, qs))
    if len(edges) < 3:
        return None
    edges[0] = min(edges[0], s.min())
    edges[-1] = max(edges[-1], s.max())
    return edges

def try_sort(df: pd.DataFrame, by_pref=("lcb","support","lift")):
    if df is None or len(df)==0:
        return df
    by = [c for c in by_pref if c in df.columns]
    if not by:
        return df
    return df.sort_values(by, ascending=[False] + [False]*(len(by)-1)).reset_index(drop=True)

def find_patterns(df: pd.DataFrame, min_support:int=10, topk:int=25, max_single_for_pairs:int=30):
    # ----- baseline
    y = (df["outcome"]=="win").astype(int).to_numpy()
    baseline = float(y.mean())
    base_logit = logit(baseline)

    # ----- feature partition
    exclude = {"outcome","preempted","enter","entry","level","symbol","ts","outcome_ts","r1_ts","tp_ts"}
    bin_cols, num_cols = [], []
    for c in df.columns:
        if c in exclude: 
            continue
        s = df[c]
        if detect_binary_series(s, c):
            bin_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)

    # ----- singletons
    sing_rows = []
    for c in bin_cols:
        s = df[c].fillna(0).astype(int)
        n1 = int(s.sum())
        if n1 < min_support:
            continue
        k1 = int(((s==1) & (df["outcome"]=="win")).sum())
        wr1 = k1 / n1
        lcb1 = wilson_lcb(k1, n1)
        lift = wr1 / baseline if baseline>0 else float("nan")
        dlog = logit(wr1) - base_logit
        sing_rows.append([c, f"{c}=1", n1, k1, wr1, lcb1, lift, dlog])
    single_df = safe_df(sing_rows)
    single_df = try_sort(single_df)
    if len(single_df) > topk:
        single_df = single_df.head(topk)

    # ----- pairs from top singles
    pair_rows = []
    top_single_feats = list(single_df["feature"].head(max_single_for_pairs)) if "feature" in single_df.columns else []
    for a, b in combinations(top_single_feats, 2):
        s = (df[a].fillna(0).astype(int)==1) & (df[b].fillna(0).astype(int)==1)
        n = int(s.sum())
        if n < max(min_support, 20):
            continue
        k = int((s & (df["outcome"]=="win")).sum())
        wr = k/n
        lcb = wilson_lcb(k, n)
        lift = wr / baseline if baseline>0 else float("nan")
        dlog = logit(wr) - base_logit
        pair_rows.append([f"{a} & {b}", f"{a}=1 AND {b}=1", n, k, wr, lcb, lift, dlog])
    pairs_df = safe_df(pair_rows)
    pairs_df = try_sort(pairs_df)
    if len(pairs_df) > topk:
        pairs_df = pairs_df.head(topk)

    # ----- numeric bins
    num_rows = []
    for c in num_cols[:50]:  # cap for speed
        edges = quantile_edges(df[c], q=10)
        if edges is None:
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
        bins = pd.cut(s, bins=edges, include_lowest=True)
        grp = df.groupby(bins, observed=False)["outcome"].agg(
            wins=lambda x: (x=="win").sum(),
            support="size"
        ).reset_index().rename(columns={"outcome":"bin"})
        grp = grp[grp["support"] >= min_support]
        if len(grp)==0:
            continue
        grp["wr"] = grp["wins"] / grp["support"]
        grp["lcb"] = [wilson_lcb(int(w), int(n)) for w,n in zip(grp["wins"], grp["support"])]
        grp["lift"] = grp["wr"] / baseline if baseline>0 else float("nan")
        grp["delta_logit"] = [logit(float(w)) - base_logit for w in grp["wr"]]
        grp["feature"] = c
        grp["pattern"] = grp.iloc[:,0].astype(str)
        num_rows.append(grp[["feature","pattern","support","wins","wr","lcb","lift","delta_logit"]])
    num_df = (pd.concat(num_rows, ignore_index=True)
              if len(num_rows) else safe_df([], COLUMNS))
    num_df = try_sort(num_df)
    if len(num_df) > topk:
        num_df = num_df.head(topk)

    return baseline, single_df, pairs_df, num_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="outputs/BTCUSDT/events_all.parquet", help="Path to merged events")
    ap.add_argument("--outdir", default="outputs/patterns", help="Output directory")
    ap.add_argument("--min_support", type=int, default=10, help="Minimum resolves per pattern")
    ap.add_argument("--topk", type=int, default=25, help="Rows per table")
    args = ap.parse_args()

    if not os.path.exists(args.events):
        raise FileNotFoundError(f"{args.events} not found. Hint: run merge_events.py first.")

    df = pd.read_parquet(args.events)

    # Focus on resolved, non-preempted
    if "preempted" in df.columns:
        df = df[~df["preempted"].fillna(False)]
    df = df[df["outcome"].isin(["win","loss"])].copy()

    n_total = len(df)
    if n_total == 0:
        raise SystemExit("No resolved rows (win/loss) after filtering; nothing to analyze.")

    baseline, singles, pairs, numeric = find_patterns(df, min_support=args.min_support, topk=args.topk)

    os.makedirs(args.outdir, exist_ok=True)
    paths = {}
    def dump(name, data):
        p = os.path.join(args.outdir, f"{name}.csv")
        (data if data is not None else pd.DataFrame(columns=COLUMNS)).to_csv(p, index=False)
        paths[name] = p

    dump("top_singleton", singles)
    dump("top_pairs", pairs)
    dump("top_numeric", numeric)

    # Combined shortlist with source tag
    parts = []
    if singles is not None and len(singles): 
        t = singles.copy(); t["source"]="singleton"; parts.append(t)
    if pairs is not None and len(pairs):
        t = pairs.copy(); t["source"]="pair"; parts.append(t)
    if numeric is not None and len(numeric):
        t = numeric.copy(); t["source"]="numeric"; parts.append(t)
    shortlist = (pd.concat(parts, ignore_index=True)
                 if parts else pd.DataFrame(columns=COLUMNS+["source"]))
    shortlist.to_csv(os.path.join(args.outdir, "shortlist.csv"), index=False)

    print(json.dumps({
        "baseline_wr": round(float(baseline), 4),
        "n_resolved_rows": int(n_total),
        "outputs": paths
    }, indent=2))

if __name__ == "__main__":
    main()
