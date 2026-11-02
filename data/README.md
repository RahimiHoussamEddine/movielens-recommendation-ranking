# Dataset acquisition

Run `make fetch` to download the stable MovieLens 1M archive from its official
GroupLens record. The script verifies the pinned checksum before safely
extracting `ratings.dat` and `movies.dat` under `data/processed/ml-1m/`.

`users.dat` is not used. Raw ratings, movie metadata, and generated model
parameters are ignored by Git in keeping with the upstream no-redistribution
terms. The project commits only aggregate audit, evaluation, and report files.
See [`NOTICE.md`](../NOTICE.md) and the upstream linked README for the full data
terms.
# Data handling

Run `make fetch` to retrieve MovieLens 1M from GroupLens. If the upstream host
is unavailable, the script uses the mirror documented by Microsoft's
Recommenders project. Both paths must match SHA-256
`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` before
extraction. The downloaded archive and extracted files stay under `data/raw/`
and are ignored by Git.

The archive contains `ratings.dat`, `movies.dat`, `users.dat`, and the source
README. The analysis loads only ratings and movie metadata; it does not load
demographics. It writes aggregate audit counts, metrics, and plots to `reports/`
and stores the local inference bundle under `artifacts/v1/`. The bundle and
source data are not committed. See [`../NOTICE.md`](../NOTICE.md) for terms.
