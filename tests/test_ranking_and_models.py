import numpy as np
from scipy.sparse import csr_matrix

from movielens_recommendation_ranking.models import fit_item_similarity
from movielens_recommendation_ranking.ranking import evaluate_scores


def test_item_similarity_has_removed_diagonal_and_is_symmetric():
    positive = csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32))
    similarity = fit_item_similarity(positive, shrinkage=0)

    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 0)
    assert np.all(similarity >= 0)


def test_evaluation_excludes_seen_items_and_finds_future_positive():
    seen = csr_matrix(np.array([[1, 1, 0, 0]], dtype=np.float32))
    scores = np.array([[100.0, 90.0, 0.8, 0.2]], dtype=np.float32)
    movie_ids = np.array([10, 20, 30, 40], dtype=np.int32)
    popularity = np.array([0, 0, 1, 1], dtype=np.float32)
    genres = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    evaluation = evaluate_scores(
        scores,
        seen,
        {0: {2}},
        movie_ids,
        popularity,
        genres,
        cutoffs=(1, 2),
    )

    assert evaluation["recommendations"][0].tolist() == [2, 3]
    assert evaluation["metrics"]["ndcg@1"] == 1.0
    assert evaluation["metrics"]["catalog_coverage@1"] == 0.25
