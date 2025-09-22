from pathlib import Path
import json, pandas as pd
from cto10r.util import wilson_lcb

ROOT = Path("outputs/BTCUSDT")
folds = [0, 1]
wins = losses = 0
for f in folds:
    d = ROOT / f"fold_{f}"
    tg = None
    for name in ["trades_gated.csv", "trades_gated_presched.csv"]:
        p = d / name
        if p.exists():
            tg = pd.read_csv(p)
            break
    if tg is None:
        raise SystemExit(f"Missing trades for fold {f}")
    win_col = tg.get("is_win")
    loss_col = tg.get("is_loss")
    if win_col is None or loss_col is None:
        outcome = tg.get("outcome", pd.Series([], dtype=object))
        win_col = (outcome == "win").astype(int)
        loss_col = (outcome == "loss").astype(int)
    wins += int(pd.to_numeric(win_col, errors="coerce").fillna(0).sum())
    losses += int(pd.to_numeric(loss_col, errors="coerce").fillna(0).sum())

resolved = wins + losses
ppv = wins / resolved if resolved else 0.0
lcb = wilson_lcb(wins, resolved)
print(f"F0+F1: wins={wins} losses={losses} resolved={resolved} ppv={ppv:.3f} lcb={lcb:.3f}")
