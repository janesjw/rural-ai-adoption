# Figure Code

Each script rebuilds one numbered figure:

| Script | Output |
|---|---|
| `01_main_figure.py` | `Figure_1.png` |
| `02_gradient_stability.py` | `Figure_2.png` |
| `A1_composition.py` | `Figure_A1.png` |
| `A2_within_state.py` | `Figure_A2.png` |
| `A3_formation_event_study.py` | `Figure_A3.png` |
| `A4_exposure_trend_break.py` | `Figure_A4.png` |
| `A5_exposure_reconciliation.py` | `Figure_A5.png` |

Run all figures from the repository root:

```bash
python plotting/run_all.py
```

To choose another output folder:

```bash
python plotting/run_all.py --out path/to/output
```
