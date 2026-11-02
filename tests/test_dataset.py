import pandas as pd

from movielens_recommendation_ranking.dataset import temporal_split


def test_temporal_split_is_per_user_chronological_and_disjoint():
    records = []
    for user_id in (1, 2):
        for step in range(20):
            records.append(
                {
                    "user_id": user_id,
                    "movie_id": step + 1,
                    "rating": 4.0 if step % 2 == 0 else 3.0,
                    "timestamp": step // 2,
                }
            )
    ratings = pd.DataFrame(records)

    train, validation, test = temporal_split(ratings)

    assert len(train) + len(validation) + len(test) == len(ratings)
    train_pairs = set(map(tuple, train[["user_id", "movie_id"]].to_numpy()))
    validation_pairs = set(map(tuple, validation[["user_id", "movie_id"]].to_numpy()))
    test_pairs = set(map(tuple, test[["user_id", "movie_id"]].to_numpy()))
    assert not train_pairs.intersection(validation_pairs | test_pairs)
    assert not validation_pairs.intersection(test_pairs)
    for user_id in (1, 2):
        pieces = [frame.loc[frame.user_id == user_id] for frame in (train, validation, test)]
        ordered = [
            list(
                frame.sort_values(["timestamp", "movie_id"])[["timestamp", "movie_id"]].itertuples(
                    index=False, name=None
                )
            )
            for frame in pieces
        ]
        assert ordered[0][-1] < ordered[1][0]
        assert ordered[1][-1] < ordered[2][0]


def test_temporal_split_rejects_too_small_histories():
    tiny = pd.DataFrame(
        {"user_id": [1, 1], "movie_id": [1, 2], "rating": [4.0, 3.0], "timestamp": [1, 2]}
    )
    try:
        temporal_split(tiny)
    except ValueError as error:
        assert "training interaction" in str(error)
    else:
        raise AssertionError("Expected small user history to be rejected")
