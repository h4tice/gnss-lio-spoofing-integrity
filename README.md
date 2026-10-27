# LiDAR-inertial integrity monitoring against slow GNSS spoofing on UAVs

Code for detecting GNSS spoofing on UAVs that carry a LiDAR. A filter-independent **LiDAR–GNSS displacement-consistency
test** is combined with three innovation tests. Instead of switching GNSS off and on, a **continuous de-weighting** rule
inflates the GNSS covariance as the test statistics approach the alarm threshold. The repository contains the
simulator, readers for two public UAV datasets, the detectors, re-implementations of three published baselines, and
every experiment reported in the accompanying paper.

> Paper: *Detecting slow GNSS spoofing on UAVs with LiDAR-inertial displacement consistency and continuous
> de-weighting: detection limits on two open flight datasets* (submitted to GPS Solutions). The citation will be added
> after publication.

## What is in here

| Component | Description |
|---|---|
| LiDAR-inertial odometry | IEKF in the style of FAST-LIO2: point-to-plane residuals, voxel map, motion compensation |
| Navigation filter | 15 + 6 state error-state Kalman filter; LIO relative pose processed with stochastic cloning; GNSS position/velocity |
| Integrity monitor | A: windowed χ² (position), B: CUSUM, C: windowed χ² (velocity), D: LiDAR–GNSS displacement consistency |
| Transition rules | `soft` (proposed, continuous de-weighting), `hard` (exclusion), `voting`, `monitor`, `gnss_ins` (no LiDAR) |
| Baselines | Tanil et al. (2018) CI monitor, Gu et al. (2021) absolute-difference test, Alja'afreh et al. (2026) NIS + persistence + CUSUM + quarantine |
| Attacks | jump (15 m), ramp (0.8 m/s), slow drift (0.12 m/s), noise (×8), dropout; white-box adaptive attacker that stays below the threshold |
| Simulator | Open3D ray-cast LiDAR, IMU and GNSS models, three environments (campus, open field, flat featureless) |
| Real data | MARS-LVIG (DJI M300, Livox Avia, u-blox) and MUN-FRL (DJI M600, Velodyne VLP-16, simpleRTK2B) |

## Installation

Python 3.11 is recommended (Open3D supports 3.8–3.12). All packages go into a local virtual environment.

Windows (PowerShell):

```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked, first run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (simulation only, no downloads)

```bash
python -m gnss_lio_sim demo            # one flight, method comparison and a 3D replay window
python -m gnss_lio_sim all --quick     # small experiment (12 flights, a few minutes)
python -m gnss_lio_sim all             # full simulation (90 flights, ~3870 runs)
python -m gnss_lio_sim tune            # select detector parameters on the development set
```

Results are written to `results/<name>/` (`report.md`, `results.csv`, figures). All parameters are in
`config/default.yaml`; dataset-specific overrides are in `config/mars_lvig.yaml`, `config/mun_frl.yaml` and
`config/mun_frl_rtk.yaml`. Pass `--config <file>` to override them.

## Real data

The datasets are not included. Download them from their providers and set `real.bag_dir` / `real.cache_dir` in the
config files to your local folders.

**MARS-LVIG** (Li et al. 2024, CC BY-NC-SA 4.0): sequences `HKairport_GNSS01/02/03`, `HKisland_GNSS01/02/03`,
`AMtown03`, `AMvalley03` (`.bag` files named after the sequence).

```bash
python -m gnss_lio_sim real-lio  --seq HKairport_GNSS03                     # LiDAR-inertial odometry only
python -m gnss_lio_sim real-eval --seq HKairport_GNSS01 HKairport_GNSS02 HKairport_GNSS03 \
                                       HKisland_GNSS01 HKisland_GNSS02 HKisland_GNSS03
python -m gnss_lio_sim real-eval --seq AMtown03 AMvalley03 --out results/semireal   # semi-real group
```

Sequences whose RTK reference repeats stale positions in more than 2% of the messages are excluded automatically.
AMtown/AMvalley have no u-blox log: their experimental GNSS is the RTK trajectory plus the u-blox error series
measured on HKairport_GNSS01. Copy `results/real/HKairport_GNSS01_gnss_err.npz` to `results/semireal/` first.

**MUN-FRL** (Thalagala et al. 2024, CC BY 4.0): the drone sequences quarry1 (`flight_dataset2.bag`), quarry2
(`flight_dataset3.bag`) and lighthouse (`flight_dataset5_fixed.bag`), together with their `<seq>.pos` (single-point)
and `<seq>_ppk.pos` (PPK reference) files. quarry1 is used for calibration only.

```bash
python -m gnss_lio_sim real-lio  --config config/mun_frl.yaml --seq quarry1 lighthouse quarry2 --out results/munfrl
python -m gnss_lio_sim real-eval --config config/mun_frl.yaml     --seq lighthouse quarry2 --out results/munfrl
python -m gnss_lio_sim real-eval --config config/mun_frl_rtk.yaml --seq lighthouse quarry2 --out results/munfrl_rtk
```

The MUN-FRL reader (`gnss_lio_sim/real/mun_frl.py`) handles several properties of the full-length recordings:

- No static transforms are recorded, so the LiDAR–IMU extrinsics come from the dataset's M600 calibration.
- The recording computer's clock is wrong; it is aligned to UTC with NMEA time-of-day and an RTK–PPK position fit.
- The LiDAR time stamp is the scan end in one flight and the scan start in the others.
- The IMU time stamps are rebuilt from the sensor clock.

## Analyses in the paper

Run these after the simulation and real-data evaluations above. Add `--groups real munfrl ...` to re-run only some
data groups.

```bash
python -m gnss_lio_sim baselines tune   # tune the baselines on the development set (same rule as the proposed method)
python -m gnss_lio_sim baselines run    # closed-loop comparison           -> results/baselines/
python -m gnss_lio_sim roc              # detection-only ROC / AUC         -> results/roc/
python -m gnss_lio_sim drift-sweep      # slow-drift rate sweep            -> results/drift_sweep/
python -m gnss_lio_sim adaptive         # adaptive (sub-threshold) attacker -> results/adaptive/
python -m gnss_lio_sim sensitivity      # LIO exclusion-bound sensitivity  -> results/sensitivity/
python paper/make_figures.py            # English figures of the paper     -> paper/figures/
```

`results_summary/` contains the summary tables behind the paper's tables, so the reported numbers can be checked
without re-running the experiments.

## Main results

The proposed method is shown against the strongest alternative.

| Data | False alarms | Slow drift (0.12 m/s) detected before 5 m | Slow-drift AUC (proposed / innovation-based) | Worst-case error, adaptive attacker |
|---|---|---|---|---|
| MARS-LVIG (standalone u-blox) | 0% | 63% | 0.99 / 0.52–0.54 | 6.2 m (others 8.2–105 m) |
| MUN-FRL single-point | 0.25% | 92% | 1.00 / 0.49–0.63 | 8.0 m (others 15.6–89 m) |
| MUN-FRL RTK-fixed | 3.25% | 100% | 0.92 / 0.51–0.55 | 3.5 m (others 5.7–110 m) |

**Known limitations**

- In flat, featureless terrain the LiDAR-based tests degrade, and a GNSS/INS monitor performs better (simulation).
- With very precise GNSS (RTK) or fast flight, the speed-independent LIO covariance leads to false alarms above 2%.
- Jumps are detected in about 0.9 s, slower than the NIS monitor of Alja'afreh et al. (2026).

## Repository layout

```
config/                  default parameters and dataset-specific overrides
gnss_lio_sim/
  sim/                   environments, trajectories, sensor models, flight simulation
  real/                  MARS-LVIG and MUN-FRL readers, real-data LIO and flight records, reports
  lio.py                 LiDAR-inertial odometry (IEKF)
  ins.py, geometry.py    error-state filter core, SO(3)
  fusion.py              GNSS / LIO / IMU fusion
  detection.py           integrity tests and transition rules
  spoofing.py            attack injection and the adaptive attacker
  baselines.py           Tanil 2018, Gu 2021, Alja'afreh 2026 re-implementations
  roc.py, drift_sweep.py, adaptive.py, sensitivity.py   analyses
  experiment.py, tuning.py, metrics.py, analysis.py     experiment runner, tuning, metrics, reports
  replay3d.py, rosbag_export.py                         3D replay, ROS 2 bag export
paper/make_figures.py    figures of the paper
results_summary/         summary tables behind the paper
```

Code comments and generated reports are partly in Turkish.

## Datasets and references

- Li H, et al. (2024) MARS-LVIG dataset. Int J Robot Res 43:1114–1127. https://doi.org/10.1177/02783649241227968
- Thalagala RG, et al. (2024) MUN-FRL. Int J Robot Res 43:1853–1866. https://doi.org/10.1177/02783649241238358
- Tanil C, et al. (2018) IEEE Trans Aerosp Electron Syst 54:131–143. https://doi.org/10.1109/TAES.2017.2739924
- Gu N, Xing F, You Z (2021) Sensors 21:6769. https://doi.org/10.3390/s21206769
- Alja'afreh M, Karime A, Al Mallah R (2026) Drones 10:738. https://doi.org/10.3390/drones10100738
- Xu W, et al. (2022) FAST-LIO2. IEEE Trans Robot 38:2053–2073. https://doi.org/10.1109/TRO.2022.3141876

## License

MIT (see `LICENSE`). The datasets keep their own licenses.
