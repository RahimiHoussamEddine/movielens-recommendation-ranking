"""Full-catalog top-k ranking metrics and user-level uncertainty estimates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from movielens_recommendation_ranking.dataset import POSITIVE_RATING_THRESHOLD


def build_targets(
    holdout: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> dict[int, set[int]]:
    positives = holdout.loc[holdout["rating"] >= POSITIVE_RATING_THRESHOLD]
    targets: dict[int, set[int]] = {}
    for user_id, movie_id in positives[["user_id", "movie_id"]].itertuples(index=False):
        user_index = user_to_index[int(user_id)]
        movie_index = movie_to_index[int(movie_id)]
        targets.setdefault(user_index, set()).add(movie_index)
    return targets


def _rank_one(
    scores: np.ndarray,
    seen_indices: np.ndarray,
    movie_ids: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    row = scores.astype(np.float32, copy=True)
    row[seen_indices] = -np.inf
    available = np.flatnonzero(np.isfinite(row))
    take = min(k, len(available))
    if take == 0:
        return np.empty(0, dtype=np.int32)
    candidate_indices = (
        available
        if take == len(available)
        else available[np.argpartition(row[available], -take)[-take:]]
    )
    order = np.lexsort((movie_ids[candidate_indices], -row[candidate_indices]))
    return candidate_indices[order].astype(np.int32, copy=False)


def evaluate_scores(
    scores: np.ndarray,
    seen: csr_matrix,
    targets: dict[int, set[int]],
    movie_ids: np.ndarray,
    item_popularity: np.ndarray,
    genre_matrix: np.ndarray,
    *,
    cutoffs: Sequence[int] = (10, 20),
) -> dict[str, Any]:
    """Evaluate full-catalog recommendations and retain only aggregate-safe arrays."""
    if scores.shape != seen.shape or scores.shape[1] != len(movie_ids):
        raise ValueError("Score matrix, seen matrix, and movie IDs must align.")
    if not targets:
        raise ValueError("No positive holdout targets are available for evaluation.")
    maximum_k = max(cutoffs)
    evaluation_users = np.asarray(sorted(targets), dtype=np.int32)
    per_user: dict[str, list[float]] = {}
    recommendations: list[np.ndarray] = []
    recommended_union: dict[int, set[int]] = {int(k): set() for k in cutoffs}
    novelty_base = max(float(item_popularity.sum()), 1.0) + len(item_popularity)

    for user_index in evaluation_users:
        start, end = seen.indptr[user_index : user_index + 2]
        seen_indices = seen.indices[start:end]
        ranked = _rank_one(scores[user_index], seen_indices, movie_ids, k=maximum_k)
        recommendations.append(ranked)
        relevant = targets[int(user_index)]
        for cutoff in cutoffs:
            chosen = ranked[:cutoff]
            chosen_set = set(map(int, chosen))
            recommended_union[int(cutoff)].update(chosen_set)
            hits = np.fromiter((int(item) in relevant for item in chosen), dtype=np.float64)
            precision = float(hits.sum() / cutoff)
            recall = float(hits.sum() / len(relevant))
            discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
            dcg = float((hits * discounts).sum())
            ideal_length = min(len(relevant), cutoff)
            idcg = float((1.0 / np.log2(np.arange(2, ideal_length + 2))).sum())
            ndcg = dcg / idcg if idcg else 0.0
            hit_positions = np.flatnonzero(hits) + 1
            average_precision = (
                float(np.mean(np.arange(1, len(hit_positions) + 1) / hit_positions))
                if len(hit_positions)
                else 0.0
            )
            average_precision /= min(len(relevant), cutoff)
            prefix = f"@{cutoff}"
            per_user.setdefault(f"precision{prefix}", []).append(precision)
            per_user.setdefault(f"recall{prefix}", []).append(recall)
            per_user.setdefault(f"ndcg{prefix}", []).append(ndcg)
            per_user.setdefault(f"map{prefix}", []).append(average_precision)
            if cutoff == 10:
                novelty = -np.log2((item_popularity[chosen] + 1.0) / novelty_base)
                per_user.setdefault("novelty@10", []).append(float(novelty.mean()))
                selected_genres = genre_matrix[chosen]
                genre_count = np.count_nonzero(selected_genres.sum(axis=0))
                available_genres = max(1, np.count_nonzero(genre_matrix.sum(axis=0)))
                per_user.setdefault("genre_coverage@10", []).append(
                    float(genre_count / available_genres)
                )
                if len(chosen) > 1:
                    norms = np.linalg.norm(selected_genres, axis=1)
                    denominator = np.outer(norms, norms)
                    similarities = np.divide(
                        selected_genres @ selected_genres.T,
                        denominator,
                        out=np.zeros_like(denominator),
                        where=denominator > 0,
                    )
                    pair_mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
                    diversity = 1.0 - float(similarities[pair_mask].mean())
                else:
                    diversity = 0.0
                per_user.setdefault("genre_diversity@10", []).append(diversity)

    per_user_arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in per_user.items()
    }
    aggregate = {name: float(values.mean()) for name, values in per_user_arrays.items()}
    aggregate.update(
        {
            f"catalog_coverage@{cutoff}": float(
                len(recommended_union[int(cutoff)]) / len(movie_ids)
            )
            for cutoff in cutoffs
        }
    )
    return {
        "metrics": aggregate,
        "per_user": per_user_arrays,
        "evaluation_users": evaluation_users,
        "recommendations": recommendations,
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int = 1_000,
) -> tuple[float, float]:
    if len(values) == 0 or replicates < 1:
        raise ValueError("Bootstrap needs at least one user and one replicate.")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, len(values), size=len(values))
        samples[replicate] = values[indices].mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def bootstrap_paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    replicates: int = 1_000,
) -> tuple[float, float]:
    if len(first) != len(second) or len(first) == 0:
        raise ValueError("Paired bootstrap inputs must be non-empty and equally sized.")
    difference = first - second
    return bootstrap_mean_interval(difference, seed=seed, replicates=replicates)


def activity_segment_summary(
    evaluation: dict[str, Any],
    training_positive_counts: np.ndarray,
) -> pd.DataFrame:
    users = evaluation["evaluation_users"]
    activity = training_positive_counts[users]
    labels = np.empty(len(activity), dtype=object)
    ordered = np.argsort(activity, kind="stable")
    for label, indexes in zip(
        ("Lower activity", "Mid activity", "Higher activity"),
        np.array_split(ordered, 3),
        strict=True,
    ):
        labels[indexes] = label
    ndcg = evaluation["per_user"]["ndcg@10"]
    recall = evaluation["per_user"]["recall@10"]
    rows = []
    for label in ("Lower activity", "Mid activity", "Higher activity"):
        mask = labels == label
        rows.append(
            {
                "activity_segment": label,
                "users": int(mask.sum()),
                "mean_train_positive_interactions": float(activity[mask].mean()),
                "ndcg@10": float(ndcg[mask].mean()),
                "recall@10": float(recall[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
