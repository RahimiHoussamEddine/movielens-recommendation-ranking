"""Small local API for the MovieLens title-seeded recommender."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from movielens_recommendation_ranking.bundle import load_bundle, recommend_from_titles

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "artifacts" / "v1" / "model_bundle.npz"
app = FastAPI(title="MovieLens Recommendation API", version="1.0.0")


class RecommendationRequest(BaseModel):
    titles: list[str] = Field(min_length=1, max_length=10)
    strategy: Literal["Popularity", "Item-item KNN", "Genre profile", "Truncated SVD"] = (
        "Item-item KNN"
    )
    limit: int = Field(default=10, ge=1, le=50)


@lru_cache(maxsize=1)
def get_bundle():
    return load_bundle(BUNDLE_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "MovieLens 1M local bundle"}


@app.post("/recommend")
def recommend(request: RecommendationRequest) -> dict[str, object]:
    try:
        frame = recommend_from_titles(
            get_bundle(), request.titles, strategy=request.strategy, limit=request.limit
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"strategy": request.strategy, "recommendations": frame.to_dict(orient="records")}
