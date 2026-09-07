"""Run every portable figure script in this folder inside the current interpreter.

This avoids Spyder subprocess issues and exposes the original traceback if one
individual figure fails.

Usage:
    python plotting/run_all.py
    python plotting/run_all.py --out my_output_folder
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import traceback

from _common import DEFAULT_OUTPUT_DIR


SCRIPTS = [
    "01_main_figure.py",
    "02_gradient_stability.py",
    "A1_composition.py",
    "A2_within_state.py",
    "A3_formation_event_study.py",
    "A4_exposure_trend_break.py",
    "A5_exposure_reconciliation.py",
]


def run_figure(path: Path, output_dir: Path) -> None:
    """Load one figure module and call its main function in this interpreter."""
    module_name = f"_local_plot_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    original_argv = sys.argv
    try:
        sys.argv = [str(path), "--out", str(output_dir)]
        module.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild every current paper figure.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for rebuilt PNG files.")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent

    for script in SCRIPTS:
        print(f"\n--- {script} ---")
        try:
            run_figure(script_dir / script, args.out)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
