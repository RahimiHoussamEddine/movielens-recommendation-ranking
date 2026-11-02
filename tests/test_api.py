import json

import numpy as np
from api import main


def test_health_and_recommend_handlers(monkeypatch):
    bundle = {
        "movie_ids": np.array([1, 2, 3]),
        "titles": np.array(["Alpha (2000)", "Beta (2001)", "Gamma (2002)"]),
        "genres": np.array(["Drama", "Comedy", "Drama"]),
        "genre_matrix": np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
        "item_similarity": np.array(
            [[0, 0.1, 0.9], [0.1, 0, 0.2], [0.9, 0.2, 0]], dtype=np.float32
        ),
        "item_factors": np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32),
        "popularity": np.array([4, 3, 2], dtype=np.float32),
        "genre_names": np.array(["Drama", "Comedy"]),
        "configuration": {},
    }
    monkeypatch.setattr(main, "get_bundle", lambda: bundle)

    health = main.health()
    response = main.recommend(
        main.RecommendationRequest(titles=["Alpha (2000)"], strategy="Item-item KNN", limit=2)
    )

    assert health["status"] == "ok"
    assert json.loads(json.dumps(response))["strategy"] == "Item-item KNN"
    assert [row["movie_id"] for row in response["recommendations"]] == [3, 2]
