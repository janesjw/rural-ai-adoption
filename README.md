# The Rural-Urban Divide in Business Use of Artificial Intelligence

This repository is the public data and code replication package for the Applied Economics Letters submission *The Rural-Urban Divide in Business Use of Artificial Intelligence: Evidence from U.S. States*. It is a frozen research snapshot dated 6 September 2026.

## Repository contents

- `analysis/`: portable code that rebuilds the current BTOS state-level results.
- `plotting/`: one Python script for each main-text and appendix figure.
- `data/raw/`: public-source snapshots needed by the main analysis and retained appendix inputs.
- `data/derived/`: analysis-ready results used by tables and figures.
- `figures/`: final publication figures in PNG format.
- `DATA_SOURCES.md`: official source pages, vintages, and download notes.

## Reproduction

Use Python 3.11 or later from the repository root:

```bash
python -m pip install -r requirements.txt
python analysis/build_main_results.py
python plotting/run_all.py
python analysis/verify_package.py
```

The analysis command rebuilds the current state-level analysis files in `data/derived/`. The plotting command rebuilds all seven figures in `figures/rebuilt/`. The verification command checks source-file hashes, the data vintage, headline estimates, and final image properties. The historical business-formation exercises in Online Appendix G use larger source extracts; the compact publication inputs required for Figures A3-A5 are retained in `data/derived/`.

## Current empirical snapshot

The analysis uses 20 BTOS waves, cycles 202524-202617, and averages cycles 202610-202617 for the headline cross-section. Reported AI use averages 22.9%, 21.5%, 19.2%, and 17.9% across increasing quartiles of state rurality. The raw slope is -0.098 (heteroskedasticity-robust t = -2.92); after additive industry and firm-size adjustments, it is -0.074 (t = -2.63), so 75.8% of the raw gradient remains. These estimates are descriptive state-level associations, not firm-level causal effects.

## Citation and reuse

See `CITATION.cff` for citation metadata. Code is released under the MIT License. Third-party data remain subject to their source agencies' terms.
