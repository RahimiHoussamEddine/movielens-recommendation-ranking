# MovieLens Personalization & Ranking

A reproducible recommendation-system study that compares popularity, item-item
collaborative filtering, genre profiles, and low-rank matrix factorization on a
chronological MovieLens holdout. The project reports ranking quality, catalog
coverage, novelty, and user-activity segments instead of relying on rating
prediction error alone.

## Run locally

Python 3.11–3.12 and a CPU are sufficient. The repository's runtime and
development requirements are pinned. The commands use a project-local `.venv`
when it exists.

```bash
python3 -m venv .venv
make install-dev
make fetch
make build
make verify
make test
make quality
make app
```

The dataset download is about 6 MB. `make fetch` gets the archive from GroupLens,
or from the documented Microsoft Recommenders mirror if the origin is
unavailable, verifies its SHA-256, and extracts it under `data/`, which is
ignored by Git. `make build` audits the data, creates per-user temporal splits,
tunes on validation data only, evaluates the final models on a later test
window, and generates the aggregate report.
The locally generated `artifacts/v1/model_bundle.npz` powers favorite-title
recommendations but is not committed. Only aggregate metrics and plots are
checked in.

The local FastAPI app is available with `make api`; `GET /health` checks the
process and `POST /recommend` accepts favorite titles, a ranking strategy, and
a result limit. Start the Streamlit demo with `make app` after the model bundle
has been built.

## Study design

MovieLens 1M is a stable benchmark released in 2003. The analysis treats ratings
of 4 stars or higher as positive feedback, reserves each user's latest ratings
for chronological validation/test, and compares popularity with personalized
collaborative and genre-based ranking. All metrics use the same eligible users
and unseen-candidate catalog; test labels are not used for tuning.

The dataset reflects users and titles from 2000–2003, not current viewing
preferences or catalogs. Results do not measure online engagement, causal
impact, or performance on new users. See the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
and [results](reports/RESULTS.md).

## Dataset terms

GroupLens permits research use, prohibits redistribution without separate
permission, and prohibits commercial or revenue-bearing use without permission.
Raw data and the local model bundle stay out of Git; only aggregate metrics and
figures are committed. Read [`NOTICE.md`](NOTICE.md) before using the dataset.
