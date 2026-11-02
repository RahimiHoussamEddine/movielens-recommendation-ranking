"""CLI entry point for the complete MovieLens study."""

from pathlib import Path

from movielens_recommendation_ranking.pipeline import run_study

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = run_study(
        data_dir=ROOT / "data" / "raw" / "extracted" / "ml-1m",
        output_dir=ROOT / "reports",
        bundle_path=ROOT / "artifacts" / "v1" / "model_bundle.npz",
    )
    print(f"Built results for {result['test_positive_users']:,} test users.")
    print(f"Report: {ROOT / 'reports' / 'RESULTS.md'}")
