"""MovieLens 1M parsing, source audit, and per-user chronological splitting."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

POSITIVE_RATING_THRESHOLD = 4.0
EXPECTED_RATING_COUNT = 1_000_209
EXPECTED_USER_COUNT = 6_040
EXPECTED_MOVIE_COUNT = 3_883
EXPECTED_ARCHIVE_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_movielens(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read only ratings and movie metadata; demographics are deliberately unused."""
    ratings_path = data_root / "ratings.dat"
    movies_path = data_root / "movies.dat"
    if not ratings_path.is_file() or not movies_path.is_file():
        raise FileNotFoundError("Run `make fetch` to extract the MovieLens 1M archive first.")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": "int32", "title": "string", "genres": "string"},
        encoding="latin-1",
    )

    if len(ratings) != EXPECTED_RATING_COUNT:
        raise ValueError(f"Expected {EXPECTED_RATING_COUNT:,} ratings; found {len(ratings):,}.")
    if ratings.isna().any().any() or movies.isna().any().any():
        raise ValueError("MovieLens source contains missing required values.")
    if ratings["user_id"].nunique() != EXPECTED_USER_COUNT:
        raise ValueError("MovieLens user count differs from the pinned source expectation.")
    if len(movies) != EXPECTED_MOVIE_COUNT or movies["movie_id"].nunique() != EXPECTED_MOVIE_COUNT:
        raise ValueError("MovieLens movie count differs from the pinned source expectation.")
    if movies.duplicated("movie_id").any():
        raise ValueError("Duplicate movie metadata IDs were found.")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user/movie ratings were found.")
    if not ratings["rating"].between(1.0, 5.0).all():
        raise ValueError("Ratings must be between one and five stars.")
    if not set(ratings["movie_id"].unique()).issubset(set(movies["movie_id"])):
        raise ValueError("Some rated movie IDs have no movie metadata.")

    counts = Counter(ratings["rating"].astype(float))
    timestamps = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    archive_path = data_root.parents[1] / "ml-1m.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(
            "The checksum-verified source archive is missing; run `make fetch`."
        )
    archive_hash = sha256_file(archive_path)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"MovieLens archive SHA-256 mismatch: {archive_hash}")
    audit = {
        "archive_sha256": archive_hash,
        "rating_count": len(ratings),
        "user_count": int(ratings["user_id"].nunique()),
        "movie_count": int(movies["movie_id"].nunique()),
        "ratings_per_user_min": int(ratings.groupby("user_id").size().min()),
        "duplicate_user_movie_pairs": 0,
        "rating_counts": {f"{rating:g}": int(counts[rating]) for rating in sorted(counts)},
        "positive_threshold": POSITIVE_RATING_THRESHOLD,
        "positive_rating_share": float((ratings["rating"] >= POSITIVE_RATING_THRESHOLD).mean()),
        "timestamp_start_utc": timestamps.min().isoformat(),
        "timestamp_end_utc": timestamps.max().isoformat(),
        "movie_genres": sorted(
            {
                genre
                for genre_list in movies["genres"].str.split("|")
                for genre in genre_list
                if genre != "(no genres listed)"
            }
        ),
        "demographic_file_used": False,
    }
    return ratings, movies, audit


def temporal_split(
    ratings: pd.DataFrame,
    *,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions in timestamp order, with stable ID tie-breaks."""
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    required = {"user_id", "movie_id", "rating", "timestamp"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"Ratings are missing columns: {sorted(required - set(ratings.columns))}")

    ordered = ratings.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _user_id, user_rows in ordered.groupby("user_id", sort=True):
        row_count = len(user_rows)
        test_count = max(1, math.ceil(row_count * test_fraction))
        validation_count = max(1, math.ceil(row_count * validation_fraction))
        if row_count - test_count - validation_count < 1:
            raise ValueError("Every user needs at least one training interaction after holdout.")
        train_end = row_count - test_count - validation_count
        validation_end = row_count - test_count
        train_parts.append(user_rows.iloc[:train_end])
        validation_parts.append(user_rows.iloc[train_end:validation_end])
        test_parts.append(user_rows.iloc[validation_end:])

    return tuple(
        pd.concat(parts, ignore_index=True) for parts in (train_parts, validation_parts, test_parts)
    )
