"""Sparse user-item matrix builders and collaborative/content model fits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from movielens_recommendation_ranking.dataset import POSITIVE_RATING_THRESHOLD


def build_matrices(
    train: pd.DataFrame,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> tuple[csr_matrix, csr_matrix, dict[int, int], dict[int, int]]:
    user_to_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    movie_to_index = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    rows = train["user_id"].map(user_to_index).to_numpy(dtype=np.int32)
    columns = train["movie_id"].map(movie_to_index).to_numpy(dtype=np.int32)
    shape = (len(user_ids), len(movie_ids))
    seen = csr_matrix((np.ones(len(train), dtype=np.float32), (rows, columns)), shape=shape)
    positive_mask = train["rating"].to_numpy() >= POSITIVE_RATING_THRESHOLD
    positive = csr_matrix(
        (
            np.ones(int(positive_mask.sum()), dtype=np.float32),
            (rows[positive_mask], columns[positive_mask]),
        ),
        shape=shape,
    )
    return positive, seen, user_to_index, movie_to_index


def build_genre_matrix(movies: pd.DataFrame, movie_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    genres = sorted(
        {
            genre
            for values in movies["genres"].str.split("|")
            for genre in values
            if genre != "(no genres listed)"
        }
    )
    genre_to_index = {genre: index for index, genre in enumerate(genres)}
    movie_to_genres = movies.set_index("movie_id")["genres"]
    matrix = np.zeros((len(movie_ids), len(genres)), dtype=np.float32)
    for movie_index, movie_id in enumerate(movie_ids):
        for genre in str(movie_to_genres.loc[int(movie_id)]).split("|"):
            if genre in genre_to_index:
                matrix[movie_index, genre_to_index[genre]] = 1.0
    return matrix, genres


def fit_item_similarity(positive: csr_matrix, *, shrinkage: float) -> np.ndarray:
    if shrinkage < 0:
        raise ValueError("Item similarity shrinkage must be non-negative.")
    cooccurrence = (positive.T @ positive).toarray().astype(np.float32, copy=False)
    support = cooccurrence.copy()
    counts = np.diag(cooccurrence).copy()
    denominator = np.sqrt(np.outer(counts, counts), dtype=np.float32)
    cosine = np.divide(
        cooccurrence,
        denominator,
        out=np.zeros_like(cooccurrence),
        where=denominator > 0,
    )
    if shrinkage:
        cosine *= support / (support + shrinkage)
    np.fill_diagonal(cosine, 0.0)
    return cosine


def fit_svd(
    positive: csr_matrix,
    *,
    components: int,
    seed: int,
) -> tuple[TruncatedSVD, np.ndarray, np.ndarray]:
    maximum = min(positive.shape) - 1
    if not 1 <= components <= maximum:
        raise ValueError(f"SVD components must be between 1 and {maximum}.")
    estimator = TruncatedSVD(
        n_components=components,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    )
    user_factors = estimator.fit_transform(positive).astype(np.float32, copy=False)
    item_factors = estimator.components_.T.astype(np.float32, copy=False)
    return estimator, user_factors, item_factors


def score_models(
    positive: csr_matrix,
    item_popularity: np.ndarray,
    item_similarity: np.ndarray,
    genre_matrix: np.ndarray,
    *,
    svd_user_factors: np.ndarray,
    svd_item_factors: np.ndarray,
) -> dict[str, np.ndarray]:
    item_scores = np.asarray(positive @ item_similarity, dtype=np.float32)
    user_genres = np.asarray(positive @ genre_matrix, dtype=np.float32)
    genre_scores = user_genres @ genre_matrix.T
    svd_scores = svd_user_factors @ svd_item_factors.T
    return {
        "Popularity": np.broadcast_to(
            item_popularity.astype(np.float32, copy=False), item_scores.shape
        ),
        "Item-item KNN": item_scores,
        "Genre profile": genre_scores,
        "Truncated SVD": svd_scores,
    }


def model_configuration(
    *,
    shrinkage: float,
    components: int,
    seed: int,
    positive_interactions: int,
) -> dict[str, Any]:
    return {
        "positive_rating_threshold": POSITIVE_RATING_THRESHOLD,
        "positive_training_interactions": positive_interactions,
        "item_similarity": "binary positive co-occurrence cosine with support shrinkage",
        "item_similarity_shrinkage": shrinkage,
        "genre_model": "user profile of liked training genres; binary genre features",
        "matrix_factorization": "TruncatedSVD on binary positive user-item matrix",
        "svd_components": components,
        "svd_random_state": seed,
    }
