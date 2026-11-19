# Validation report

## Protocol

The run used seed 18 and a deterministic visual-group split. Model
selection compared class-weighted multinomial logistic regression with a
class-balanced 300-tree random forest on the 727-crop validation split. Both
models used the same 130 color and texture features. Controlled dark, bright,
and warm-color copies augmented training data. After selection, the random
forest was refit on training plus validation crops and evaluated once on 355
held-out crops from 218 photographs.

## Model selection

| Candidate | Validation accuracy | Balanced accuracy | Macro F1 | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 98.49% | 98.50% | 98.33% | 0.0851 |
| Random forest | 98.76% | 98.74% | 98.75% | 0.1549 |

Random forest won by validation macro F1, with log loss used as a secondary
criterion.

## Test results

| Metric | Result |
| --- | ---: |
| Accuracy | 1.0000 |
| Balanced accuracy | 1.0000 |
| Macro F1 | 1.0000 |
| Ordinal expected-stage MAE | 0.0345 stages |
| Multiclass log loss | 0.0302 |
| Automatic review rate | 0.0225 |

Confusion matrix with rows as references and columns as predictions, ordered
fresh/normal/rotten:

```text
[[ 84,   0,   0],
 [  0, 211,   0],
 [  0,   0,  60]]
```

The model made no class-decision errors on this internal test split. Eight
correctly classified crops crossed the confidence/entropy review threshold.
Error-review recall is therefore undefined, and the current review policy still
requires evaluation on external errors.

## Controlled image-condition stress tests

The untouched test images were digitally transformed after training. The same
transform families were used for training augmentation.

| Transformation | Accuracy | Macro F1 | Ordinal MAE | Review rate |
| --- | ---: | ---: | ---: | ---: |
| Brightness × 0.7 | 1.0000 | 1.0000 | 0.0402 | 0.0225 |
| Brightness × 1.3 | 1.0000 | 1.0000 | 0.0420 | 0.0197 |
| Warm RGB scale 1.12/1.00/0.88 | 1.0000 | 1.0000 | 0.0386 | 0.0197 |

These are synthetic covariate-shift checks. They do not substitute for external
camera, cultivar, market, or longitudinal evaluation.

## Interpretation

The internal split demonstrates that the runnable pipeline learns the source
labels with few same-dataset errors. It does not establish field reliability.
Bounding-box crops, dataset-specific backgrounds, annotation rules, and closely
related acquisitions may make this task easier than user photographs. No
external dataset was evaluated, and no shelf-life outcome was available.

The split audit found zero exact byte-duplicate groups and zero decoded-pixel
duplicate groups across partitions. It flagged 14 test crops as visual-similarity
candidates using a difference-hash distance of at most 2 and resized RGB MSE of
at most 0.0025. This audit is intentionally conservative and does not prove that
the remaining images represent independent biological fruit.

Machine-readable measurements are in `reports/validation.json`. Metrics in this
document were copied from that completed run and are not paper benchmark claims.

