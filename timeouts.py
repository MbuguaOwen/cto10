# timeouts.py
import os, sys, math
import pandas as pd

ROOT = r"outputs/BTCUSDT"   # change if needed

def detect_unit(ts_max: int) -> str:
    # Seconds ~ 1.7e9 vs millis ~ 1.7e12
    return "ms" if ts_max and ts_max > 10**11 else "s"

def load_events(ev_path: str) -> pd.DataFrame:
    df = pd.read_parquet(ev_path)
    if "ts" not in df.columns or "outcome" not in df.columns:
        raise ValueError(
            f"{ev_path} missing required columns. Found: {list(df.columns)}; "
            "expected at least ['ts','outcome']"
        )
    unit = detect_unit(int(df["ts"].max()))
    dt = pd.to_datetime(df["ts"], unit=unit, utc=True)
    df["ym"] = dt.dt.strftime("%Y-%m")
    # Derive booleans from outcome
    oc = df["outcome"].astype(str).str.lower()
    df["is_win"] = oc.eq("win")
    df["is_loss"] = oc.eq("loss")
    df["resolved"] = df["is_win"] | df["is_loss"]
    return df

def summarize(df: pd.DataFrame, fold: int) -> pd.DataFrame:
    g = df.groupby("ym", observed=True).agg(
        n=("resolved", "size"),
        w=("is_win", "sum"),
        l=("is_loss", "sum"),
        res=("resolved", "mean"),
    )
    g["timeouts"] = g["n"] - (g["w"] + g["l"])
    g["timeout_rate"] = (g["timeouts"] / g["n"]).round(3).fillna(0.0)
    resolved = (g["w"] + g["l"]).replace(0, pd.NA)
    g["ppv"] = (g["w"] / resolved).round(3)
    g["fold"] = fold
    # order columns nicely
    return g.reset_index()[["fold","ym","n","res","timeout_rate","w","l","ppv"]]

def main():
    if not os.path.isdir(ROOT):
        print(f"Not found: {ROOT}")
        sys.exit(1)

    rows = []
    for name in sorted(os.listdir(ROOT)):
        if not name.startswith("fold_"): 
            continue
        try:
            fold = int(name.split("_")[-1])
        except ValueError:
            continue
        ev_path = os.path.join(ROOT, name, "events.parquet")
        if not os.path.exists(ev_path):
            print(f"[fold {fold}] events.parquet missing, skipping")
            continue
        try:
            df = load_events(ev_path)
            rows.append(summarize(df, fold))
        except Exception as e:
            print(f"[fold {fold}] ERROR reading {ev_path}: {e}")
            continue

    if not rows:
        print("No event summaries found.")
        return

    out = pd.concat(rows, ignore_index=True)
    # Pretty print
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(out.to_string(index=False))

    # Quick totals
    tot = out.assign(
        resolved=lambda d: d["w"] + d["l"],
        timeouts=lambda d: d["n"] - (d["w"] + d["l"])
    )
    N = int(tot["n"].sum())
    R = int(tot["resolved"].sum())
    T = int(tot["timeouts"].sum())
    ppv = (tot["w"].sum() / R) if R else float("nan")
    print("\n=== Overall ===")
    print(f"events={N:,}  resolved={R:,} ({R/N:.1%})  timeouts={T:,} ({T/N:.1%})  PPV_resolved={ppv:.3f}" if N else "No events")

if __name__ == "__main__":
    main()
