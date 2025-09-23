import argparse, json, random
from pathlib import Path
import pandas as pd
random.seed(1337)

# Assumes per-fold training table in outputs/<SYM>/fold_k/artifacts/train_*.csv or .parquet
def _find_train_table(fold_dir: Path):
    arts = fold_dir / "artifacts"
    cands = list(arts.glob("train_*.parquet")) + list(arts.glob("train_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No train_* table under {arts}")
    p = cands[0]
    if p.suffix == ".csv": df = pd.read_csv(p)
    else: df = pd.read_parquet(p)
    return p, df

def permute_labels(fold: int, sym="BTCUSDT"):
    fd = Path(f"outputs/{sym}/fold_{fold}")
    p, df = _find_train_table(fd)
    # unify binary labels; expected columns: is_win, is_loss
    if "is_win" not in df.columns or "is_loss" not in df.columns:
        raise ValueError("Expected is_win/is_loss columns in train table")
    lab = df["is_win"].astype(int).to_numpy().copy()
    random.shuffle(lab)
    df["is_win"]  = lab
    df["is_loss"] = 1 - df["is_win"]
    out = p.with_name(p.stem + "_PERMUTED" + p.suffix)
    (df.to_csv if out.suffix==".csv" else df.to_parquet)(out, index=False)
    print(f"[permute] wrote {out}")

def shift_labels(fold: int, days: int, sym="BTCUSDT"):
    fd = Path(f"outputs/{sym}/fold_{fold}")
    p, df = _find_train_table(fd)
    if "ymd" not in df.columns or "is_win" not in df.columns or "is_loss" not in df.columns:
        raise ValueError("Need ymd, is_win, is_loss in train table")
    df = df.sort_values("ymd").copy()
    for col in ("is_win","is_loss"):
        s = df[col].astype(int).to_numpy()
        # +days forward shift (labels refer to the future)
        import numpy as np
        shifted = np.roll(s, days)
        if days > 0: shifted[:days] = 0
        else:       shifted[days:] = 0
        df[col] = shifted
    out = p.with_name(p.stem + f"_SHIFT{days}d" + p.suffix)
    (df.to_csv if out.suffix==".csv" else df.to_parquet)(out, index=False)
    print(f"[shift] wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("permute"); p1.add_argument("--fold", type=int, required=True)
    p2 = sub.add_parser("shift");   p2.add_argument("--fold", type=int, required=True); p2.add_argument("--days", type=int, default=1)
    args = ap.parse_args()
    if args.cmd=="permute": permute_labels(args.fold)
    elif args.cmd=="shift": shift_labels(args.fold, args.days)
