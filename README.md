# AMR Docking Simulator (2D)

Robust 2D differential-drive docking simulator with sensor corruption, filtering/gating, docking FSM, and safety supervisor with dynamic obstacles.

## Quick Start

### Phase 1 (clean GT)
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/run_single.py
```

### Phase 2 (sensor corruption + robust stack)
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/run_single.py \
  --use_sensor --sigma_d 0.05 --sigma_yaw 0.1 --p_drop 0.2 --p_out 0.05 \
  --outlier_dist 0.6 --outlier_yaw_deg 25 \
  --use_filter --use_gating --hold_last --show_meas
```

### Phase 3 (FSM + Safety + Dynamic Obstacles)
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/run_single.py \
  --use_sensor --use_filter --use_gating --hold_last \
  --use_fsm --use_safety --scenario crossing --show_meas
```

## Fixed Test Cases + Noise Calibration
- Test cases are fixed once and reused for all evaluations:
  - `outputs/test_cases.csv`
- Noise strength is calibrated to target baseline success rate range:
  - `outputs/noise_calibration.json`

## Official Benchmark Splits (Locked)
These are the **official benchmark splits** used for tuning and final evaluation:
- Train: `outputs/train_cases.csv` (420 cases)
- Val: `outputs/val_cases.csv` (90 cases)
- Test holdout: `outputs/test_cases_holdout.csv` (90 cases)

**Important:** `src/test_cases.generate_test_cases()` will **not overwrite** existing files unless explicitly forced.
To regenerate, delete the CSV(s) you want to replace.

To (re)generate:
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/calibrate_noise.py --target_min 0.4 --target_max 0.7
```

## Phase 4 Evaluation (Ablations + Safety)
Run the full evaluation pipeline:
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/run_evaluation.py
python /Users/M1/Desktop/CMM/AMR_Docking/make_eval_plots.py
```

Outputs:
- `outputs/phase4_ablation.csv`
- `outputs/phase4_safety.csv`
- `outputs/phase4_ablation_plots.png`
- `outputs/phase4_safety_plots.png`

## Phase 5 Auto-Tuning
Random search over controller/FSM/robustness parameters (uses fixed cases + calibrated noise).
Latency is treated as a **fixed stress setting** via `--latency_steps` (default: 5 steps).

Example:
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/tune_params.py \
  --trials 300 --seed 42 --max_cases 200 \
  --use_sensor --use_filter --use_gating --hold_last --use_fsm --use_safety \
  --scenario mixed --latency_steps 5 \
  --load_noise outputs/noise_calibration.json \
  --load_cases outputs/train_cases.csv \
  --val_cases outputs/val_cases.csv
```

Outputs:
- `outputs/tuning_results.csv`
- `outputs/tuning_topk.csv`
- `outputs/best_config.json`
- `outputs/tuning_summary.md`

### Safety Metrics Note
- `phase4_safety.csv` includes **collision-free metrics**:
  - `safe_success_rate` = success without any collision
  - `safe_steps_mean` = mean steps among collision-free successes
- Phase 6 recovery metrics (if enabled in evaluation):
  - `recovery_mean` = mean recovery episodes per trial
  - `deadlock_rate` = fraction of trials that hit deadlock

## Phase 6 Recovery (Re-try Logic)
Enable recovery in single-run demos:
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/run_single.py \
  --use_sensor --use_filter --use_gating --hold_last \
  --use_fsm --use_safety --use_recovery --scenario crossing
```
Default recovery behavior: wait → rotate → backoff → re-approach, with max retries.
- Safety evaluation uses a **mixed obstacle scenario** (crossing + short cut-in + persistent cut-in) for fair comparison.

## Phase 5–6 Evaluation (Holdout)
Use the **test holdout set** for final reporting:
```bash
python /Users/M1/Desktop/CMM/AMR_Docking/run_evaluation_phase56.py \
  --cases outputs/test_cases_holdout.csv --max_cases 0
python /Users/M1/Desktop/CMM/AMR_Docking/make_phase56_plots.py
```

## Key Files
- `src/dynamics.py` — differential-drive kinematics
- `src/actuation.py` — actuation noise (slip-like)
- `src/test_cases.py` — fixed test case generation/loading
- `src/sensor.py` — GT relative pose + noise/dropout/latency/outliers
- `src/filtering.py` — EMA filter
- `src/robust.py` — gating + hold_last + stale handling
- `src/fsm.py` — docking FSM
- `src/safety.py` — safety supervisor
- `src/obstacles.py` — dynamic obstacle scenarios
- `run_single.py` — interactive/demo runner
- `run_phase3_demo.py` — static Phase 3 demo plot
- `run_phase3_batch.py` — batch safety evaluation
- `calibrate_noise.py` — noise calibration helper
- `run_evaluation.py` — full ablation + safety evaluation
- `make_eval_plots.py` — plots from evaluation CSVs

## Notes
- Actuation noise defaults: `sigma_v=0.03`, `sigma_w=0.05` (3%/5%).
- Cut-in obstacles can be persistent; safety may stop indefinitely in those cases.
