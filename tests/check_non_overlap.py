import pathlib as p
import sys

import pandas as pd


def main() -> int:
    root = p.Path("outputs") / "BTCUSDT"
    fails = 0
    for fold_dir in sorted(root.glob("fold_*")):
        fp = fold_dir / "trades_gated.csv"
        if not fp.exists():
            print(f"[WARN] missing {fp}")
            continue
        df = pd.read_csv(fp)
        need = {"ts", "end_ts"}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            print(f"[FAIL] {fp} missing columns {missing}")
            fails += 1
            continue
        df = df.sort_values("ts").reset_index(drop=True)
        overlap = (df["ts"].shift(-1) < df["end_ts"]).fillna(False)
        n_overlap = int(overlap.sum())
        if n_overlap > 0:
            print(f"[FAIL] {fp} overlaps={n_overlap}")
            fails += 1
        else:
            print(f"[OK] {fp} non-overlapping")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
