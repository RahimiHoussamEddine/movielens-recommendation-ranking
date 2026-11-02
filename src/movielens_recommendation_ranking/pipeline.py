"""Train, tune, evaluate, report, and locally package the recommendation study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommendation_ranking.bundle import save_bundle
from movielens_recommendation_ranking.dataset import load_movielens, temporal_split
from movielens_recommendation_ranking.models import (
    build_genre_matrix,
    build_matrices,
    fit_item_similarity,
    fit_svd,
    model_configuration,
    score_models,
)
from movielens_recommendation_ranking.ranking import (
    activity_segment_summary,
    bootstrap_mean_interval,
    bootstrap_paired_difference,
    build_targets,
    evaluate_scores,
)
from movielens_recommendation_ranking.reporting import write_report

SEED = 2026
SHRINKAGE_GRID = (5.0, 20.0, 50.0)
SVD_COMPONENT_GRID = (16, 32, 64)
BOOTSTRAP_REPLICATES = 1_000


def _split_summary(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    def boundaries(frame: pd.DataFrame, *, first: bool) -> pd.DataFrame:
        ordered = frame.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
        selected = (
            ordered.groupby("user_id", sort=True).head(1)
            if first
            else ordered.groupby("user_id", sort=True).tail(1)
        )
        return selected.set_index("user_id")[["timestamp", "movie_id"]]

    train_last = boundaries(train, first=False)
    validation_first = boundaries(validation, first=True)
    validation_last = boundaries(validation, first=False)
    test_first = boundaries(test, first=True)
    common = train_last.index.intersection(validation_first.index).intersection(test_first.index)
    time_order = bool(
        all(
            tuple(train_last.loc[user]) < tuple(validation_first.loc[user])
            and tuple(validation_last.loc[user]) < tuple(test_first.loc[user])
            for user in common
        )
        and len(common) == train["user_id"].nunique()
    )
    return {
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "users": {
            "train": int(train["user_id"].nunique()),
            "validation": int(validation["user_id"].nunique()),
            "test": int(test["user_id"].nunique()),
        },
        "temporal_order_check": time_order,
    }


def _fit_scores(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    *,
    shrinkage: float,
    components: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    positive, seen, user_to_index, movie_to_index = build_matrices(train, user_ids, movie_ids)
    genre_matrix, genre_names = build_genre_matrix(movies, movie_ids)
    popularity = np.asarray(positive.sum(axis=0)).ravel().astype(np.float32)
    item_similarity = fit_item_similarity(positive, shrinkage=shrinkage)
    _svd, user_factors, item_factors = fit_svd(positive, components=components, seed=SEED)
    scores = score_models(
        positive,
        popularity,
        item_similarity,
        genre_matrix,
        svd_user_factors=user_factors,
        svd_item_factors=item_factors,
    )
    model = {
        "positive": positive,
        "seen": seen,
        "popularity": popularity,
        "item_similarity": item_similarity,
        "genre_matrix": genre_matrix,
        "genre_names": genre_names,
        "item_factors": item_factors,
    }
    mappings = {"user_to_index": user_to_index, "movie_to_index": movie_to_index}
    return scores, model, mappings


def _evaluate_all(
    scores: dict[str, np.ndarray],
    seen: Any,
    targets: dict[int, set[int]],
    movie_ids: np.ndarray,
    popularity: np.ndarray,
    genre_matrix: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    evaluations: dict[str, dict[str, Any]] = {}
    safe: dict[str, Any] = {}
    for name, matrix in scores.items():
        evaluated = evaluate_scores(matrix, seen, targets, movie_ids, popularity, genre_matrix)
        evaluations[name] = evaluated
        safe[name] = {"metrics": evaluated["metrics"]}
    return evaluations, safe


def run_study(data_dir: Path, output_dir: Path, bundle_path: Path) -> dict[str, Any]:
    ratings, movies, data_audit = load_movielens(data_dir)
    train, validation, test = temporal_split(ratings)
    split = _split_summary(train, validation, test)
    if not split["temporal_order_check"]:
        raise ValueError("Temporal split ordering check failed.")

    user_ids = np.sort(ratings["user_id"].unique()).astype(np.int32)
    movie_ids = np.sort(movies["movie_id"].unique()).astype(np.int32)
    base_positive, base_seen, user_to_index, movie_to_index = build_matrices(
        train, user_ids, movie_ids
    )
    genre_matrix, genre_names = build_genre_matrix(movies, movie_ids)
    validation_targets = build_targets(validation, user_to_index, movie_to_index)
    if not validation_targets:
        raise ValueError("No positive validation interactions are available.")
    validation_popularity = np.asarray(base_positive.sum(axis=0)).ravel().astype(np.float32)

    tuning: dict[str, list[dict[str, float | int]]] = {
        "item_similarity": [],
        "truncated_svd": [],
    }
    best_shrinkage, best_item_ndcg = SHRINKAGE_GRID[0], -np.inf
    for shrinkage in SHRINKAGE_GRID:
        similarity = fit_item_similarity(base_positive, shrinkage=shrinkage)
        item_scores = np.asarray(base_positive @ similarity, dtype=np.float32)
        evaluated = evaluate_scores(
            item_scores,
            base_seen,
            validation_targets,
            movie_ids,
            validation_popularity,
            genre_matrix,
        )
        ndcg = evaluated["metrics"]["ndcg@10"]
        tuning["item_similarity"].append({"shrinkage": shrinkage, "validation_ndcg@10": ndcg})
        if ndcg > best_item_ndcg:
            best_shrinkage, best_item_ndcg = shrinkage, ndcg

    best_components, best_svd_ndcg = SVD_COMPONENT_GRID[0], -np.inf
    for components in SVD_COMPONENT_GRID:
        _estimator, val_user_factors, val_item_factors = fit_svd(
            base_positive, components=components, seed=SEED
        )
        svd_scores = val_user_factors @ val_item_factors.T
        evaluated = evaluate_scores(
            svd_scores,
            base_seen,
            validation_targets,
            movie_ids,
            validation_popularity,
            genre_matrix,
        )
        ndcg = evaluated["metrics"]["ndcg@10"]
        tuning["truncated_svd"].append({"components": components, "validation_ndcg@10": ndcg})
        if ndcg > best_svd_ndcg:
            best_components, best_svd_ndcg = components, ndcg

    development = pd.concat([train, validation], ignore_index=True)
    test_targets = build_targets(test, user_to_index, movie_to_index)
    if not test_targets:
        raise ValueError("No positive test interactions are available.")

    scores, final_model, mappings = _fit_scores(
        development,
        movies,
        user_ids,
        movie_ids,
        shrinkage=best_shrinkage,
        components=best_components,
    )
    test_targets = build_targets(test, mappings["user_to_index"], mappings["movie_to_index"])
    evaluations, safe_test = _evaluate_all(
        scores,
        final_model["seen"],
        test_targets,
        movie_ids,
        final_model["popularity"],
        final_model["genre_matrix"],
    )

    popularity_user = evaluations["Popularity"]["per_user"]["ndcg@10"]
    for model_index, (name, evaluation) in enumerate(evaluations.items()):
        absolute_interval = bootstrap_mean_interval(
            evaluation["per_user"]["ndcg@10"],
            seed=SEED + model_index,
            replicates=BOOTSTRAP_REPLICATES,
        )
        difference_interval = bootstrap_paired_difference(
            evaluation["per_user"]["ndcg@10"],
            popularity_user,
            seed=SEED + 100 + model_index,
            replicates=BOOTSTRAP_REPLICATES,
        )
        safe_test[name]["ndcg@10_ci95"] = list(absolute_interval)
        safe_test[name]["ndcg@10_difference_vs_popularity_ci95"] = list(difference_interval)
        activity = activity_segment_summary(
            evaluation,
            np.asarray(final_model["positive"].sum(axis=1)).ravel(),
        )
        safe_test[name]["activity_segments"] = activity.to_dict(orient="records")

    chosen = model_configuration(
        shrinkage=best_shrinkage,
        components=best_components,
        seed=SEED,
        positive_interactions=int(final_model["positive"].nnz),
    )
    results = {
        "study": "MovieLens 1M chronological top-k recommendation",
        "seed": SEED,
        "data_audit": data_audit,
        "split": split,
        "validation_positive_users": len(validation_targets),
        "test_positive_users": len(test_targets),
        "tuning": tuning,
        "selected_configuration": chosen,
        "test": safe_test,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "catalog_size": len(movie_ids),
        "evaluation_protocol": "Full catalog excluding each user's development history; positive future ratings are relevant.",
    }

    movie_meta = movies.set_index("movie_id").loc[movie_ids]
    local_bundle = {
        "movie_ids": movie_ids,
        "titles": movie_meta["title"].astype(str).to_numpy(),
        "genres": movie_meta["genres"].astype(str).to_numpy(),
        "genre_matrix": final_model["genre_matrix"],
        "item_similarity": final_model["item_similarity"],
        "item_factors": final_model["item_factors"],
        "popularity": final_model["popularity"],
        "genre_names": np.asarray(genre_names, dtype=str),
        "configuration": chosen,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_bundle(bundle_path, local_bundle)
    write_report(results, output_dir)
    return results
