import argparse, json, math
from pathlib import Path
import pandas as pd

def dsr(sharpe, n):
    # deflated Sharpe (Bailey et al., simplified)
    from math import sqrt
    if n <= 3: return float('nan')
    gamma = 0.5772
    return (sharpe - (1 - gamma) * (math.sqrt((1 + sharpe**2) / (n - 1)))) / math.sqrt((1 - gamma**2) / (n - 1))

def pbo_from_folds(folds):
    # Probability of Backtest Overfitting from sorted folds (toy estimate)
    wr = folds["test_ppv"].rank(method="average") / len(folds)
    return float((wr > 0.5).mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="BTCUSDT")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    rows = []
    for f in args.folds:
        js = Path(f"outputs/{args.sym}/fold_{f}/gate_summary.json")
        if not js.exists(): continue
        g = json.loads(js.read_text())
        tr_ppv = g.get("ppv", float('nan')); tr_lcb = g.get("ppv_lcb", float('nan'))
        tg = Path(f"outputs/{args.sym}/fold_{f}/trades_gated.csv")
        if not tg.exists(): continue
        d = pd.read_csv(tg)
        res = int(d.get("is_win",0).sum() + d.get("is_loss",0).sum())
        wr  = float(d.get("is_win",0).sum() / res) if res else float('nan')
        rows.append(dict(fold=f, train_ppv=tr_ppv, train_lcb=tr_lcb, test_ppv=wr, resolved=res))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("no folds found"); raise SystemExit(0)
    # crude Sharpe proxy from ppv over baseline 0.5
    df["sharpe_proxy"] = (df["test_ppv"] - 0.5) / 0.1
    df["dsr"] = [dsr(s, max(r,3)) for s,r in zip(df["sharpe_proxy"], df["resolved"])]
    pbo = pbo_from_folds(df)
    print(df.to_string(index=False))
    print(f"\nPBO (toy)={pbo:.3f}")
