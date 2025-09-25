import argparse, json, numpy as np, pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/BTCUSDT", help="symbol outputs root")
    ap.add_argument("--fold", default="fold_0", help="fold dir")
    ap.add_argument("--tau", type=float, default=None, help="fixed tau; if None, use gate tau")
    ap.add_argument("--r_tol", type=float, default=1e-6, help="R equality tolerance")
    args = ap.parse_args()

    fold = Path(args.root) / args.fold
    ev = pd.read_parquet(fold / "events.parquet")
    sel_path = fold / "gate_selected.parquet"
    sim_path = fold / "sim_scheduled.parquet"
    sel = pd.read_parquet(sel_path) if sel_path.exists() else None
    sim = pd.read_parquet(sim_path) if sim_path.exists() else None

    # SAME-SIGNAL: given same feed & tau, selected vs scheduled sets match
    if sel is not None and sim is not None:
        key_col = "candidate_id" if "candidate_id" in sel.columns else "ts"
        k1 = set(sel[key_col])
        k2 = set(sim[key_col])
        extra = sorted(list(k1 - k2))[:5]
        miss = sorted(list(k2 - k1))[:5]
        same_signal = (k1 == k2)
    else:
        same_signal = None
        extra = []
        miss = []

    # SAME-TRADE: exit types and R equal within tolerance
    same_trade = None
    max_r_diff = None
    if sim is not None:
        if "R" in sim.columns and "R" in ev.columns:
            j = sim.merge(ev[["ts", "R"]], on="ts", how="left", suffixes=("_sim", ""))
            diffs = (j["R_sim"] - j["R"]).abs().dropna()
            max_r_diff = float(diffs.max()) if len(diffs) else 0.0
            same_trade = bool((diffs <= args.r_tol).all())

    print(
        json.dumps(
            {
                "same_signal": same_signal,
                "same_trade": same_trade,
                "max_r_diff": max_r_diff,
                "examples_only_in_selected": extra,
                "examples_only_in_scheduled": miss,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

