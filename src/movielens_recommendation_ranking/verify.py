"""Check that aggregate results and the local bundle are complete and aligned."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from movielens_recommendation_ranking.bundle import load_bundle

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "results.json"
BUNDLE = ROOT / "artifacts" / "v1" / "model_bundle.npz"
EXPECTED_MODELS = ("Popularity", "Item-item KNN", "Genre profile", "Truncated SVD")
FORBIDDEN_RESULT_KEYS = {
    "user_id",
    "user_ids",
    "user_index",
    "evaluation_users",
    "per_user",
    "recommendations",
}


def _assert_aggregate_only(value: object) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_RESULT_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"User-level fields cannot be stored in results: {sorted(forbidden)}")
        for nested in value.values():
            _assert_aggregate_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_aggregate_only(nested)


def verify() -> None:
    if not RESULTS.is_file():
        raise FileNotFoundError("Run `make build` before verification.")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    _assert_aggregate_only(data)
    if data["data_audit"]["rating_count"] != 1_000_209:
        raise ValueError("The rating count does not match MovieLens 1M.")
    if data["data_audit"]["demographic_file_used"] is not False:
        raise ValueError("The study must not use user demographics.")
    if not data["split"]["temporal_order_check"]:
        raise ValueError("The chronological split check did not pass.")
    if set(data["test"]) != set(EXPECTED_MODELS):
        raise ValueError("The final test report is missing a model result.")
    for model in EXPECTED_MODELS:
        metrics = data["test"][model]["metrics"]
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Non-finite aggregate metric in {model} results.")
        if len(data["test"][model]["activity_segments"]) != 3:
            raise ValueError(f"Activity breakdown is incomplete for {model}.")

    bundle = load_bundle(BUNDLE)
    n = len(bundle["movie_ids"])
    if n != data["catalog_size"] or len(bundle["titles"]) != n:
        raise ValueError("Model bundle catalog does not align with the report.")
    if bundle["item_similarity"].shape != (n, n):
        raise ValueError("Item similarity matrix has the wrong shape.")
    if not all(
        np.isfinite(bundle[name]).all()
        for name in ("item_similarity", "item_factors", "popularity")
    ):
        raise ValueError("Model bundle contains non-finite values.")
    report = ROOT / "reports" / "RESULTS.md"
    figures = (
        ROOT / "reports" / "figures" / "test_quality_and_coverage.png",
        ROOT / "reports" / "figures" / "activity_segments.png",
    )
    if not report.is_file() or any(
        not path.is_file() or path.stat().st_size == 0 for path in figures
    ):
        raise FileNotFoundError("A generated report or figure is missing.")
    print(
        "Verification passed: source audit, aggregate metrics, plots, and local model bundle align."
    )


if __name__ == "__main__":
    verify()
