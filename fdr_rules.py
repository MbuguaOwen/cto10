import argparse, pandas as pd
from pathlib import Path

def bh_fdr(pvals, alpha=0.10):
    p = pd.Series(pvals).sort_values().reset_index(drop=True)
    m = len(p); thresh = (pd.Series(range(1,m+1))/m) * alpha
    keep = p <= thresh
    k = keep[keep].index.max() if keep.any() else -1
    cutoff = float(p.iloc[k]) if k>=0 else 0.0
    return cutoff

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--sym", default="BTCUSDT")
    args = ap.parse_args()
    f = Path(f"outputs/{args.sym}/fold_{args.fold}/rules_eval.csv")
    df = pd.read_csv(f)
    # assume df has p_value or 1-ppv proxy; adapt if needed
    p = df["p_value"] if "p_value" in df.columns else (1.0 - df["ppv_clipped"])
    cut = bh_fdr(p, alpha=args.alpha)
    survivors = df.loc[p <= cut]
    print(f"[BH-FDR] alpha={args.alpha} cutoff={cut:.4g} survivors={len(survivors)}/{len(df)}")
    out = f.with_name("rules_eval_fdr.csv")
    survivors.to_csv(out, index=False); print(f"wrote {out}")
