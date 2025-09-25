cto10r — Robust, parity-faithful research pipeline (Quick Ignition on, Vol Gate off)

This repo runs a walkforward research pipeline for pattern discovery and ML gating on crypto ticks/bars.
The current configuration mirrors the “quick ignition” live policy:

Non-overlap: enabled

Policy: quick (one ticket per side per symbol within a short “quick window”; defaults to 6h)

Volatility prefilter: disabled (no atr_p cutoff)

Labeling: strict first-touch on ticks (causal), with clean separation of administrative vs real timeouts

Simulation: fees + (optional) funding applied; no inventory overlaps needed because quick policy blocks stacking

1) What’s in here

cto10r/ – pipeline code:

walkforward.py – stages: features → tick_labeling → mining → simulate

ticks.py – tick-window labeling (first-touch TP/SL/timeout, no peeking)

bars.py, candidates.py – features & candidate generation (CUSUM thinning)

mining.py – rule mining with Wilson LCB & optional FDR

gate.py, ml.py – ML gating, calibration, coverage/LCB targeting

util.py – schema & helpers

configs/

motifs.yaml – main configuration (see highlights below)

inputs/ (you create)

bars_1m/{SYMBOL}/{SYMBOL}-1m-{YYYY}-{MM}.csv

ticks/{SYMBOL}/{SYMBOL}-ticks-{YYYY}-{MM}.csv

outputs/ – per-symbol, per-fold artifacts (features, candidates.parquet, events.parquet, mining & sim results)

2) Requirements & installation

Python 3.10–3.11 recommended

pip install -r requirements.txt (pandas, numpy, scikit-learn, pyarrow/fastparquet, tqdm, etc.)

Windows PowerShell note: use $env:VAR=value to set env vars temporarily in a shell.

3) Data layout
inputs/
  bars_1m/
    BTCUSDT/BTCUSDT-1m-2025-04.csv
  ticks/
    BTCUSDT/BTCUSDT-ticks-2025-04.csv


configs/motifs.yaml controls symbol list and month range.

4) Config highlights (current defaults)
candidates:
  mode: blind
  blind:
    thinning:
      type: cusum_by_ret
      params:
        ret_lookback: 360
        k_sigma: 1.6
        drift: 0.0
        min_gap_bars: 0          # no candidate-level collapse
        fallback_stride_k: 0
    sides: both
  vol_prefilter: null            # ✅ VOL GATE OFF (no atr_p filter)

labels:
  r_mult: 5
  horizon_hours: 480
  non_overlap: true
  non_overlap_policy: quick      # ✅ QUICK IGNITION (busy for quick window)
  quick_ignition_hours: 6
  intra_bar_epsilon_ms: 1
  min_risk_atr_frac: 0.001
  eta_atr: 15.0
  eta_by_atr_p:                  # ✅ sane η (TP reachable)
    - [0, 35, 0.50]
    - [35, 65, 1.00]
    - [65, 85, 2.00]
    - [85, 95, 4.00]
    - [95, 101, 6.00]
  eta_cap: { min: 0.50, max: 6.00 }
  time_scale_from_eta: { base_eta: 0.30, clamp: [0.75, 10.0] }
  stop_band_atr: 0.05
  no_ticks_policy: skip          # data gaps = administrative (excluded from learning)

mining.rules:
  auto_binning: true
  n_bins_cont: 5
  max_terms: 2
  top_k: 60
  min_support_frac: 0.001
  min_months: 2
  min_wins: 2
  wilson_z: 1.96
  fdr: { enabled: true, alpha: 0.10 }

execution_sim:
  fees:
    maker_bps: 1.0
    taker_bps: 5.0
    entry_is_maker: false
    exit_is_maker: true
  funding:
    enabled: true
    rate_per_hour: 0.0001
    sign: 1.0
  inventory:
    enable: false               # stacking prevented by quick window


Why this shape?

Quick ignition reproduces your previously better win/loss balance and avoids stacked signals from one impulse.

Vol filter is removed to avoid upstream bias; instead, we dedupe one-per-side per quick window at the candidate stage (below).

5) Pipeline stages
A) Features

Causal features (trend/ATR/Donchian/nonlinear, plus age features). Percentiles fit on train only.

python -m cto10r --config configs/motifs.yaml --mode walkforward --only features --force

B) Candidate generation & quick-window dedupe

CUSUM triggers create raw candidates.

Before labeling, we dedupe to one candidate per side per quick window (e.g., 6h). This dramatically reduces downstream administrative preemption while exactly matching the quick policy.

You’ll see logs like:

[SYMBOL] dedupe[quick=6h]: kept X/Y (Z%)

C) Tick labeling (first-touch)

For each candidate, build a tick window [t_entry+ε, deadline] and resolve:

Long: TP if price ≥ target first; SL if price ≤ stop first

Short: TP if price ≤ target first; SL if price ≥ stop first

Timeout: neither hit by deadline

Administrative: no_ticks_policy: skip marks such rows as preempted=True (excluded from learning)

Diagnostics:

[ticks] sanity: geom_bad=… tp_any=… sl_any=… first_touch_wins=… first_touch_losses=… timeouts=…
[SYMBOL] events: N (wins=W, losses=L, timeouts=T)
[SYMBOL] events detail: preempted=P, timeouts_nonpreempt=T-P


Run:

python -m cto10r --config configs/motifs.yaml --mode walkforward --only tick_labeling --force


Debug speed knob: limit labeling via env var:

PowerShell: $env:LABEL_MAX=5000

Bash: LABEL_MAX=5000
Then run tick_labeling to test logic quickly.

D) Mining

Bins literals, computes Wilson lower confidence bounds, applies optional FDR.

Promotes top rules and identifies loser masks.

python -m cto10r --config configs/motifs.yaml --mode walkforward --only mining --force

E) Simulation

Gating (ML/LCB) → scheduled trades.

Quick ignition policy is respected because we dedup upstream; no stacking occurs.

Fees & funding applied.

python -m cto10r --config configs/motifs.yaml --mode walkforward --only simulate --force

6) Live-parity modes (how to switch later)

Quick ignition (current): non_overlap: true, non_overlap_policy: quick, quick_ignition_hours: 6, plus candidate quick-window dedupe.

Busy until exit: non_overlap_policy: to_exit (block new entries until the previous trade exits). Remove/adjust candidate dedupe (it’s longer than quick).

Allow overlaps: non_overlap: false, non_overlap_policy: none + inventory caps in sim. (Not used now.)

Consistency across config + code is required to pass Same-signal / Same-trade tests.

7) Troubleshooting & red flags

“Wins=0, losses huge” during labeling

Likely TP unreachable: revert to sane η (see config above) and verify first-touch logic.

Check tp_any in the sanity line; if near zero → TP too far or geometry wrong.

Lots of timeouts

If timeouts_nonpreempt is high, consider slightly longer horizon_hours (e.g., 600) or adjust r_mult.

If preempted is high and no_ticks_windows>0 → data gaps; no_ticks_policy: skip is correct (excluded).

ValueError: invalid literal for int() with base 10: 'auto'

Ensure any n_bins_cont: auto is only used where the code supports it; otherwise set a concrete int (e.g., 5).

FileNotFoundError: candidates.parquet

Run stages in order; features must precede tick_labeling; candidates are created in features stage.

Too few promotions (promoted=0)

You may simply be sample-starved after strict filtering. Increase train months or relax rule thresholds. (Optional small-sample fallback can be added if desired.)

8) Performance tips

Quick-window dedupe upstream reduces candidate count dramatically → faster labeling + cleaner training.

Use LABEL_MAX while iterating on labeler logic.

Keep walkforward.progress: true to monitor long runs.

9) Metrics & definitions

PPV: wins / (wins + losses) among resolved (timeouts excluded).

LCB: Wilson lower confidence bound at wilson_z (default 1.96 ≈ 95%).

Coverage: fraction of candidates selected by the gate (or scheduled) over total candidates.

10) Typical run
# Clean start after config/code changes
python -m cto10r --config configs/motifs.yaml --mode walkforward --clean --force --only features

# Label
python -m cto10r --config configs/motifs.yaml --mode walkforward --only tick_labeling --force

# Mine & simulate
python -m cto10r --config configs/motifs.yaml --mode walkforward --only mining --force
python -m cto10r --config configs/motifs.yaml --mode walkforward --only simulate --force


Expected logs (per fold):

Candidate quick dedupe line

Tick sanity line

Events summary with non-zero wins

Mining promotions summary

Simulate coverage/PPV/LCB

11) Contributing / extending

Add curated motifs (whitelists) by introducing a curated_motifs: block in the YAML and wiring them into mining.py if you want hand-picked literals.

To compare parity modes, keep separate YAMLs (e.g., motifs_quick.yaml, motifs_to_exit.yaml) and re-run walkforward.

12) License

Choose an appropriate license for your project (e.g., MIT/Apache-2.0) and place it in LICENSE.

Final checklist before you run

 vol_prefilter is null (or removed)

 non_overlap_policy: quick and quick_ignition_hours: 6

 Quick-window dedupe is active in stage_tick_labeling

 Labeler uses first-touch (TP/SL) with t_entry + ε start

 eta_by_atr_p is 0.5–6.0 band

 no_ticks_policy: skip

 Fees/funding set to match your venue