# sentence-polarity-baselines

[![ci](https://github.com/MoustafaMZaalouk/sentence-polarity-baselines/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MoustafaMZaalouk/sentence-polarity-baselines/actions/workflows/ci.yml)

Small, reproducible comparison of linear text classifiers on the Pang & Lee sentence polarity dataset (10,662 movie-review snippets, balanced). It looks at accuracy, calibration (ECE) and the most confident mistakes. CPU only, runs in about 15 seconds.

This is an AI-assisted educational project written by a student. It is not production work, not client work, and the numbers are not a benchmark claim.

## Results (5 random 80/20 splits, 10662 sentences)

| model | accuracy (mean) | std over seeds | pooled bootstrap 95% | ECE (mean) |
|---|---|---|---|---|
| `majority` | 0.500 | 0.000 | 0.490 to 0.509 | 0.500 |
| `nb_unigram` | 0.779 | 0.008 | 0.771 to 0.787 | 0.089 |
| `lr_word12` | 0.776 | 0.011 | 0.768 to 0.783 | 0.046 |
| `lr_word12_char` | 0.787 | 0.012 | 0.779 to 0.794 | 0.023 |
| `svm_word12_calibrated` | 0.775 | 0.009 | 0.767 to 0.783 | 0.025 |

Plots: `results/accuracy.png`. Raw numbers, per-seed accuracy and the 15 most confident errors of `lr_word12_char` (seed 0): `results/results.json`.

## How to read this
- Hyperparameters were fixed in advance and were not tuned on the test split. There is no separate validation split, so the model choice is only descriptive.
- The seed-to-seed std is the more honest uncertainty. The pooled bootstrap interval stacks test sets from different seeds that overlap, so it is somewhat optimistic.
- Differences between `nb_unigram`, `lr_word12`, `lr_word12_char` and the calibrated SVM are about one point. That is within the seed noise, so this does not show any one of them is better.
- ECE for `majority` is large because it predicts with full confidence on a balanced set. It is a sanity check, not a model.
- Many confident errors are sentences with mixed sentiment ("a mess, but a sincere mess"), where a bag of n-grams has no way to tell which clause wins.
- Run on one dataset of movie-review sentences only. Nothing here says how it would do on other text.

## Data
Not included. `make run` downloads the original archive from the Cornell movie-review-data page and checks its SHA-256 (`a05bef...ce9`) before using it. The dataset belongs to Bo Pang and Lillian Lee (Pang & Lee, ACL 2005, "Seeing stars"); see their README for terms. Code here is MIT only.

## Run
```
pip install -r requirements.txt
make test     # 7 tests; downloads the dataset once
make smoke    # 1,500-sentence, 1-seed run
make run      # full run, rewrites results/
```

## Layout
`src/polarity/` data, models, metrics, run | `tests/` | `results/` committed output of the last full run | `.github/workflows/ci.yml`

## License
MIT, code only. Copyright Moustafa Mouhamed Zaalouk.
