"""Verify the public data and code replication package."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "derived" / "latest_results.json"
MANIFEST = ROOT / "data" / "raw" / "latest_source_manifest.json"
FIGURES = [
    "Figure_1.png",
    "Figure_2.png",
    "Figure_A1.png",
    "Figure_A2.png",
    "Figure_A3.png",
    "Figure_A4.png",
    "Figure_A5.png",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    require(
        math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
        f"Expected {expected}, found {actual}.",
    )


def check_results() -> None:
    require(RESULTS.is_file(), "Run analysis/build_main_results.py first.")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    data = results["data"]
    cross = results["cross_section"]
    pooled = results["pooled_twenty"]

    require(len(data["waves"]) == 20, "Expected 20 BTOS waves.")
    require(data["waves"][0] == "202524", "Unexpected first BTOS cycle.")
    require(data["waves"][-1] == "202617", "Unexpected latest BTOS cycle.")
    require(data["latest_window"] == [f"2026{period:02d}" for period in range(10, 18)], "Unexpected headline window.")
    require(data["units"] == 51, "Expected 51 state-level units.")
    close(cross["raw_slope"], -0.09768025303686027)
    close(cross["adjusted_slope"], -0.07404376340046877)
    close(cross["adjusted_share"], 0.75802182220523)
    require(pooled["negative_waves"] == 20, "The rural slope should be negative in all 20 waves.")
    require(pooled["waves_t_below_minus_two"] == 12, "Expected 12 wave-specific robust t-statistics below -2.")


def check_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key in ("btos", "qcew", "acs", "broadband"):
        record = manifest[key]
        path = ROOT / "data" / "raw" / record["file"]
        require(path.is_file(), f"Manifest file is missing: {record['file']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        require(digest == record["sha256"], f"SHA-256 mismatch for {record['file']}.")


def check_figures() -> None:
    figure_dir = ROOT / "figures"
    actual = sorted(path.name for path in figure_dir.glob("*.png"))
    require(actual == sorted(FIGURES), f"Unexpected final figure inventory: {actual}")
    for name in FIGURES:
        with Image.open(figure_dir / name) as image:
            dpi = image.info.get("dpi", (0, 0))
            require(min(image.size) >= 1500, f"{name} is too small: {image.size}")
            require(all(abs(value - 600) < 1 for value in dpi), f"{name} is not 600 dpi: {dpi}")


def main() -> None:
    check_manifest()
    check_results()
    check_figures()
    print("PASS: source hashes, data vintage, headline estimates, and seven figures are consistent.")


if __name__ == "__main__":
    main()
