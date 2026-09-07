# Analysis

`build_main_results.py` rebuilds the current 20-wave BTOS state panel and the headline composition, within-state, complement, and diffusion outputs from the compact raw snapshots. It includes assertions for the latest BTOS period and the QCEW sector mapping so stale inputs do not silently enter the figures.

Run from the repository root:

```bash
python analysis/build_main_results.py
```
