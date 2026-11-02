import numpy as np

from movielens_recommendation_ranking.bundle import (
    load_bundle,
    recommend_from_titles,
    save_bundle,
)


def sample_bundle():
    return {
        "movie_ids": np.array([1, 2, 3, 4]),
        "titles": np.array(["Alpha (2000)", "Beta (2001)", "Gamma (2002)", "Delta (2003)"]),
        "genres": np.array(["Drama", "Comedy", "Drama", "Action"]),
        "genre_matrix": np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        "item_similarity": np.array(
            [
                [0, 0.1, 0.9, 0.2],
                [0.1, 0, 0.2, 0.8],
                [0.9, 0.2, 0, 0.1],
                [0.2, 0.8, 0.1, 0],
            ],
            dtype=np.float32,
        ),
        "item_factors": np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        "popularity": np.array([4, 3, 2, 1], dtype=np.float32),
        "genre_names": np.array(["Drama", "Comedy"]),
        "configuration": {"seed": 1},
    }


def test_bundle_round_trip_and_title_recommendation(tmp_path):
    path = tmp_path / "model_bundle.npz"
    save_bundle(path, sample_bundle())
    loaded = load_bundle(path)
    result = recommend_from_titles(loaded, ["Alpha (2000)"], strategy="Item-item KNN", limit=2)

    assert result["movie_id"].tolist() == [3, 4]
    assert 1 not in result["movie_id"].tolist()


def test_bundle_resolves_ui_title_with_id():
    result = recommend_from_titles(
        sample_bundle(), ["Beta (2001) [#2]"], strategy="Genre profile", limit=3
    )
    assert 2 not in result["movie_id"].tolist()
