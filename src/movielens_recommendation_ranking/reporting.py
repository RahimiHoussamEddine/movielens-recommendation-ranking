"""Aggregate-only report tables and publication-quality figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_ORDER = ("Popularity", "Item-item KNN", "Genre profile", "Truncated SVD")


def _markdown_table(frame: pd.DataFrame, *, digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        if pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:,}")
        else:
            display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    top10_rows = []
    top20_rows = []
    for model in MODEL_ORDER:
        item = results["test"][model]
        metrics = item["metrics"]
        low, high = item["ndcg@10_ci95"]
        diff_low, diff_high = item["ndcg@10_difference_vs_popularity_ci95"]
        top10_rows.append(
            {
                "Model": model,
                "Precision@10": metrics["precision@10"],
                "Recall@10": metrics["recall@10"],
                "NDCG@10": metrics["ndcg@10"],
                "NDCG 95% CI": f"[{low:.4f}, {high:.4f}]",
                "MAP@10": metrics["map@10"],
                "Delta NDCG CI vs popularity": f"[{diff_low:.4f}, {diff_high:.4f}]",
            }
        )
        top20_rows.append(
            {
                "Model": model,
                "Precision@20": metrics["precision@20"],
                "Recall@20": metrics["recall@20"],
                "NDCG@20": metrics["ndcg@20"],
                "MAP@20": metrics["map@20"],
            }
        )
    top10_frame = pd.DataFrame(top10_rows)
    top20_frame = pd.DataFrame(top20_rows)
    discovery_frame = pd.DataFrame(
        [
            {
                "Model": model,
                "Catalog coverage@10": results["test"][model]["metrics"]["catalog_coverage@10"],
                "Catalog coverage@20": results["test"][model]["metrics"]["catalog_coverage@20"],
                "Novelty@10": results["test"][model]["metrics"]["novelty@10"],
                "Genre coverage@10": results["test"][model]["metrics"]["genre_coverage@10"],
                "Genre diversity@10": results["test"][model]["metrics"]["genre_diversity@10"],
            }
            for model in MODEL_ORDER
        ]
    )

    coverage = [results["test"][model]["metrics"]["catalog_coverage@10"] for model in MODEL_ORDER]
    ndcg = [results["test"][model]["metrics"]["ndcg@10"] for model in MODEL_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    palette = ["#7b8794", "#176b87", "#2a9d8f", "#e76f51"]
    axes[0].bar(MODEL_ORDER, ndcg, color=palette)
    axes[0].set_title("Future positive ranking")
    axes[0].set_ylabel("Mean NDCG@10")
    axes[0].set_ylim(0, max(ndcg) * 1.16 if max(ndcg) else 1)
    axes[1].bar(MODEL_ORDER, coverage, color=palette)
    axes[1].set_title("Catalog reach")
    axes[1].set_ylabel("Fraction of catalog recommended")
    axes[1].set_ylim(0, max(coverage) * 1.16 if max(coverage) else 1)
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.savefig(figures_dir / "test_quality_and_coverage.png", dpi=180)
    plt.close(fig)

    segment_rows = []
    for model in MODEL_ORDER:
        for row in results["test"][model]["activity_segments"]:
            segment_rows.append({"Model": model, **row})
    segment_frame = pd.DataFrame(segment_rows)
    fig, axis = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    for model, color in zip(MODEL_ORDER, palette, strict=True):
        segment = segment_frame.loc[segment_frame["Model"] == model]
        axis.plot(
            segment["activity_segment"],
            segment["ndcg@10"],
            marker="o",
            linewidth=2,
            color=color,
            label=model,
        )
    axis.set_title("Ranking quality by training-history activity")
    axis.set_ylabel("Mean NDCG@10")
    axis.set_xlabel("User activity segment")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(figures_dir / "activity_segments.png", dpi=180)
    plt.close(fig)

    rows = results["split"]["rows"]
    chosen = results["selected_configuration"]
    audit = results["data_audit"]
    report = f"""# MovieLens 1M temporal recommendation results

## Main result

The test window contains the latest per-user interactions after a chronological
train/validation split. The tables report full-catalog top-10 and top-20
recommendations for users with at least one future rating of four or five stars.
Each method uses the same eligible users and excludes items already rated before
the test window. Intervals are 95% seeded paired-user bootstrap intervals.

### Top-10 ranking quality

{_markdown_table(top10_frame)}

### Top-20 ranking quality

{_markdown_table(top20_frame)}

## Catalog discovery

{_markdown_table(discovery_frame)}

![Test NDCG and catalog coverage](figures/test_quality_and_coverage.png)

The difference interval is for each model's user-level NDCG@10 minus popularity.
An interval entirely above zero is evidence of higher offline ranking quality
on this benchmark; it is not evidence of an online engagement lift.

## Activity groups

Users are divided into lower, middle, and higher activity groups by the number
of positive training interactions. The same group boundaries are used for all
models.

{_markdown_table(segment_frame[["Model", "activity_segment", "users", "mean_train_positive_interactions", "ndcg@10", "recall@10"]])}

![NDCG by activity group](figures/activity_segments.png)

## Data and experiment audit

- Source: [MovieLens 1M from GroupLens](https://grouplens.org/datasets/movielens/1m/), a stable dataset released in 2003.
- Rows: {audit["rating_count"]:,} ratings from {audit["user_count"]:,} users across {audit["movie_count"]:,} movie metadata records.
- Ratings of at least {audit["positive_threshold"]:g} stars are treated as positive feedback; positive share is {audit["positive_rating_share"]:.1%}.
- Split rows: {rows["train"]:,} train, {rows["validation"]:,} validation, {rows["test"]:,} test; chronological ordering check: {results["split"]["temporal_order_check"]}.
- Validation-selected item-neighbor shrinkage: {chosen["item_similarity_shrinkage"]:g}; SVD components: {chosen["svd_components"]}.
- Demographic `users.dat` was not loaded. Results and figures contain aggregate statistics only.

## Interpretation limits

MovieLens reflects users and titles from 2000–2003. Ratings depend on exposure
and are not a randomized measure of preference; unrated titles are unknown,
not confirmed negatives. Offline future-rating performance does not predict
current catalogs, new users, deployment engagement, or causal impact. Genre
diversity, novelty, and coverage are catalog proxies. The upstream dataset has
research-only terms; see [`NOTICE.md`](../NOTICE.md).

## Reproduction

Run `make fetch`, `make build`, and `make verify`. The raw source archive and
local recommendation bundle are ignored by Git. The bundle is needed to run
the local Streamlit demo or API and is not redistributed in this repository.
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
