# Phase 2 Summary (Robust Sensing)

## Improvements Implemented
- Gating relaxation during prolonged missing/rejected measurements via `stale_jump_scale`.
- Stale-measurement speed decay and stop policy (`stale_decay`, `stale_min_scale`, `stale_stop_steps`).

## Key Findings
- Improved policy shows the biggest gains under high dropout and high yaw noise.
- Under mild conditions, differences are small; a few low-noise cases show slight regressions.

## Random-Start Evaluation (N=120)
Baseline gating vs Improved gating + stale slowdown:
- Success rate: **0.975 → 1.000**
- Mean steps: **181.7 → 165.1** (faster)
- Reject fraction: **0.100 → 0.053** (more usable measurements)

## Noise/Dropout Sweep (80 conditions)
- Average success gain: **+0.095**
- Average steps gain: **~40 steps**
- Gains peak at high `p_drop` (0.3–0.4) and high `sigma_yaw` (0.1–0.2).

## Recommended Defaults (stability-first)
Controller:
- `v_max=0.6`, `w_max=1.5`, `k_v=1.2`, `k_w=2.0`, `yaw_slow_deg=35`

Robustness:
- `ema_dist=0.3`, `ema_yaw=0.3`
- `max_jump_dist=0.3`, `max_jump_yaw_deg=20`
- `hold_last=True`, `stale_jump_scale=0.15`
- `stale_decay=0.2`, `stale_min_scale=0.2`, `stale_stop_steps=6`

## Artifacts
- `outputs/sweep_noise_dropout.csv`
- `outputs/sweep_noise_dropout.png`
