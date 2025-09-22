from pathlib import Path
import json, pandas as pd

ROOT = Path("outputs/BTCUSDT")
FOLDS = [0,1,2,3]


def load_gate_summary(fd: Path):
    art = fd / "artifacts" / "gate_summary.json"
    legacy = fd / "gate_summary.json"
    path = art if art.exists() else legacy
    with open(path, "r") as f:
        return json.load(f)


def load_trades(fd: Path):
    p1 = fd / "trades_gated.csv"
    p2 = fd / "trades_gated_presched.csv"
    if p1.exists():
        return pd.read_csv(p1)
    if p2.exists():
        return pd.read_csv(p2)
    raise FileNotFoundError(f"[{fd.name}] no trades_gated(.csv|_presched.csv)")


rows=[]
for f in FOLDS:
    fd = ROOT / f"fold_{f}"
    gs = load_gate_summary(fd)
    tg = load_trades(fd)
    win_source = tg.get("is_win", tg.get("outcome", pd.Series([], dtype=object)))
    loss_source = tg.get("is_loss", tg.get("outcome", pd.Series([], dtype=object)))
    outcome_is_str = "outcome" in tg.columns
    win_target = "win" if outcome_is_str else 1
    loss_target = "loss" if outcome_is_str else 0
    wins = int((win_source == win_target).sum())
    losses = int((loss_source == loss_target).sum())
    resolved = wins + losses
    test_ppv = (wins / resolved) if resolved else 0.0

    def wilson_lcb(w, n, z=1.96):
        if n == 0:
            return 0.0
        p = w / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        r = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
        return max(0.0, (c - r) / d)

    rows.append(
        dict(
            fold=f,
            tau=round(gs.get("tau", 0.5), 4),
            train_ppv=round(gs.get("ppv", 0.0), 3),
            train_lcb=round(gs.get("ppv_lcb", 0.0), 3),
            train_cov=round(gs.get("coverage", 0.0), 3),
            wins=wins,
            losses=losses,
            resolved=resolved,
            test_ppv=round(test_ppv, 3),
            test_lcb=round(wilson_lcb(wins, resolved), 3),
        )
    )

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\nPASS (test_ppv ≥ train_lcb):")
print((df["test_ppv"] >= df["train_lcb"]).to_string(index=False))
