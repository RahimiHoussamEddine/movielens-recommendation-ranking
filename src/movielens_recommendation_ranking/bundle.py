"""Local aggregate model bundle and title-seeded recommendation logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STRATEGIES = ("Popularity", "Item-item KNN", "Genre profile", "Truncated SVD")


def save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        movie_ids=np.asarray(bundle["movie_ids"], dtype=np.int32),
        titles=np.asarray(bundle["titles"], dtype=str),
        genres=np.asarray(bundle["genres"], dtype=str),
        genre_matrix=np.asarray(bundle["genre_matrix"], dtype=np.float32),
        item_similarity=np.asarray(bundle["item_similarity"], dtype=np.float32),
        item_factors=np.asarray(bundle["item_factors"], dtype=np.float32),
        popularity=np.asarray(bundle["popularity"], dtype=np.float32),
        genre_names=np.asarray(bundle["genre_names"], dtype=str),
        configuration=np.asarray(json.dumps(bundle["configuration"], sort_keys=True)),
    )


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Local model bundle not found at {path}. Run `make fetch` and `make build` first."
        )
    with np.load(path, allow_pickle=False) as data:
        bundle = {name: data[name].copy() for name in data.files}
    bundle["configuration"] = json.loads(str(bundle.pop("configuration")))
    return bundle


def _resolve_title_indices(bundle: dict[str, Any], titles: list[str]) -> list[int]:
    raw_titles = bundle["titles"].tolist()
    ids = bundle["movie_ids"].tolist()
    exact: dict[str, list[int]] = {}
    display: dict[str, int] = {}
    for index, title in enumerate(raw_titles):
        exact.setdefault(title.casefold(), []).append(index)
        display[f"{title} [#{ids[index]}]"] = index

    resolved: list[int] = []
    for supplied in titles:
        value = supplied.strip()
        if value in display:
            resolved.append(display[value])
            continue
        matches = exact.get(value.casefold(), [])
        if not matches:
            raise ValueError(f"Movie title was not found: {supplied}")
        if len(matches) > 1:
            choices = ", ".join(f"{raw_titles[i]} [#{ids[i]}]" for i in matches)
            raise ValueError(f"Movie title is ambiguous; use a title with its ID: {choices}")
        resolved.append(matches[0])
    return resolved


def recommend_from_titles(
    bundle: dict[str, Any],
    titles: list[str],
    *,
    strategy: str = "Item-item KNN",
    limit: int = 10,
) -> pd.DataFrame:
    """Rank unseen catalog titles using a short list of movies the visitor likes."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Strategy must be one of: {', '.join(STRATEGIES)}")
    if not titles:
        raise ValueError("Choose at least one favorite movie title.")
    if not 1 <= limit <= 50:
        raise ValueError("Recommendation limit must be between 1 and 50.")
    selected = np.asarray(_resolve_title_indices(bundle, titles), dtype=np.int32)
    if len(np.unique(selected)) != len(selected):
        selected = np.unique(selected)

    if strategy == "Popularity":
        scores = bundle["popularity"].astype(np.float64, copy=True)
    elif strategy == "Item-item KNN":
        scores = bundle["item_similarity"][selected].mean(axis=0, dtype=np.float64)
    elif strategy == "Genre profile":
        matrix = bundle["genre_matrix"].astype(np.float64, copy=False)
        profile = matrix[selected].mean(axis=0)
        profile_norm = np.linalg.norm(profile)
        item_norms = np.linalg.norm(matrix, axis=1)
        denominator = profile_norm * item_norms
        scores = np.divide(
            matrix @ profile,
            denominator,
            out=np.zeros(len(matrix), dtype=np.float64),
            where=denominator > 0,
        )
    else:
        factors = bundle["item_factors"].astype(np.float64, copy=False)
        profile = factors[selected].mean(axis=0)
        profile_norm = np.linalg.norm(profile)
        item_norms = np.linalg.norm(factors, axis=1)
        denominator = profile_norm * item_norms
        scores = np.divide(
            factors @ profile,
            denominator,
            out=np.zeros(len(factors), dtype=np.float64),
            where=denominator > 0,
        )

    scores[selected] = -np.inf
    order = np.lexsort((bundle["movie_ids"], -scores, ~np.isfinite(scores)))
    order = [int(i) for i in order if np.isfinite(scores[i])][:limit]
    return pd.DataFrame(
        {
            "movie_id": bundle["movie_ids"][order],
            "title": bundle["titles"][order],
            "genres": bundle["genres"][order],
            "score": scores[order],
        }
    )
