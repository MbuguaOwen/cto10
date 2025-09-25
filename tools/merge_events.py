import argparse, os, pandas as pd, glob

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="e.g. outputs/BTCUSDT")
ap.add_argument("--out", required=True, help="e.g. outputs/BTCUSDT/events_all.parquet")
args = ap.parse_args()

parts = []
for p in sorted(glob.glob(os.path.join(args.root, "fold_*", "events.parquet"))):
    try:
        df = pd.read_parquet(p)
        df["fold"] = os.path.basename(os.path.dirname(p))
        parts.append(df)
    except Exception as e:
        print(f"skip {p}: {e}")
if not parts:
    raise SystemExit("no fold events found")
out = pd.concat(parts, ignore_index=True)
out.to_parquet(args.out, index=False)
print(f"Wrote {args.out} with {len(out)} rows")

