# Sources and provenance

## Dataset

- [GroupLens MovieLens 1M page](https://grouplens.org/datasets/movielens/1m/) identifies the stable benchmark as about one million ratings from about 6,000 users and 4,000 movies, released in February 2003.
- The archive README supplied inside the dataset describes 1,000,209 anonymous ratings, 6,040 users, 5-star ratings, timestamp fields, and the dataset use terms. `make fetch` does not load `users.dat` into the analysis.
- The archive is pinned by SHA-256 `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`. The same digest appears in the [TensorFlow recommendation example](https://github.com/tensorflow/examples/blob/master/lite/examples/recommendation/ml/data/example_generation_movielens.py), which names the original GroupLens URL.
- [Microsoft Recommenders dataset documentation](https://github.com/recommenders-team/recommenders/blob/main/docs/datasets.rst) documents a Hugging Face backup mirror for MovieLens when the original download fails. The mirror is used only as a transport fallback; the exact archive checksum is verified before extraction, and GroupLens terms still apply.

## Citation and use terms

Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and
Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article
19. https://doi.org/10.1145/2827872

GroupLens permits research use, requires acknowledgment in resulting
publications, bars redistribution without separate permission, and bars
commercial or revenue-bearing use without prior permission. See the archive
README and [`../NOTICE.md`](../NOTICE.md).
