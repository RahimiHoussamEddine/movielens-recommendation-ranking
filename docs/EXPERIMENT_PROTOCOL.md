# Experiment protocol

## Question

Do personalized recommenders improve top-10 discovery over a global popularity
list when evaluated on future high ratings, and what happens to catalog reach,
novelty, and performance across user-activity levels?

## Data and outcome

Use the complete, checksum-verified MovieLens 1M rating and movie files. Audit
rating ranges, duplicate user/movie pairs, timestamp order, missing values, and
movie metadata joins. A rating of at least 4 stars is the predefined positive
feedback threshold. Do not use the demographic `users.dat` file.

## Temporal split

Within each user, sort interactions by timestamp and stable movie-ID tie-break.
Reserve the latest 10% for test and the preceding 10% for validation; use all
earlier interactions for training. All candidate recommendation lists exclude
movies observed in that user's training history. Report actual row and user
counts because per-user rounding means split sizes are approximate.

## Models

Compare positive-interaction popularity, cosine item-item collaborative
filtering, genre-profile similarity, and truncated-SVD latent factors. Tune
factor count and item-neighbor shrinkage on chronological validation data only.
After selection, refit on train plus validation and evaluate the latest test
period. The popularity model is the reference; test labels are used once for
the final comparison.

## Metrics and uncertainty

Report precision, recall, NDCG, and MAP at 10 and 20; catalog coverage,
average recommendation novelty, genre coverage, intra-list genre diversity,
and NDCG by training-history activity segment. Use a seeded paired bootstrap
over eligible test users for 95% intervals on NDCG and its difference from
popularity. The test distribution is the historical MovieLens community and
does not estimate modern deployment outcomes.

## Interpretation limits

Offline future-rating ranking is not a randomized online test. Ratings encode
exposure as well as preference; unobserved items are not true negatives. The
user base and catalog date from 2000–2003. Reported novelty/coverage are catalog
proxies, not user utility. The upstream dataset is research-only and must not be
redistributed or used commercially without permission.
