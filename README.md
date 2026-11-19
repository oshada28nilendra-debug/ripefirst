# 04 — Use-First Banana Prioritizer

A focused computer-vision prototype that ranks photographed bananas by **visible
deterioration**. It assigns one of three appearance stages (`fresh`, `normal`,
`rotten`), provides class probabilities and a review flag, and optionally uses
declared storage days as a small ranking tiebreaker. Dated observations can be
stored to measure changes in the model score for the same item.

The system does not estimate safe shelf life, edibility, internal quality, or
food safety from a photograph. “Use first” means “visually prioritize for human
inspection or normal stock rotation.” A rotten-looking result is not a safety
test, and a fresh-looking result is not proof that food is safe.

Implemented and validated on **18 November 2026** for bananas only.

## Quick start

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
use-first --help
```

Run a prediction with the supplied fitted model:

```bash
use-first predict \
  --model artifacts/banana_stage_model.joblib \
  --image /path/to/one-banana-crop.jpg \
  --output outputs/prediction.json
```

The input should show one banana or one consistently photographed bunch. The
training pipeline learns from annotated banana crops, so an uncropped shop
scene is outside the tested input distribution.

## Rank several items

Create `items.csv`:

```csv
item_id,image,storage_days
banana-001,/photos/banana-001.jpg,2
banana-002,/photos/banana-002.jpg,5
banana-003,/photos/banana-003.jpg,
```

Then run:

```bash
use-first rank \
  --model artifacts/banana_stage_model.joblib \
  --items items.csv \
  --output outputs/ranking.json
```

Confident items are sorted by priority. Uncertain or low-quality images are
sent to `review_image` after the confident queue. Without storage metadata,
priority equals the 0–100 visible-deterioration score. With storage days, the
default formula is:

`priority = 0.85 × visible_score + 0.15 × min(storage_days / 7, 1) × 100`

Seven days is a bounded operational normalization constant, not a shelf-life
claim. The storage contribution can be set between 0 and 0.5 with
`--storage-weight`; the default intentionally keeps the photograph dominant.

## Longitudinal observations

Record repeated, consistently framed photographs of the same item:

```bash
use-first observe \
  --model artifacts/banana_stage_model.joblib \
  --image /photos/banana-001-day1.jpg \
  --item-id banana-001 \
  --observed-at 2026-11-18T09:00:00+05:30 \
  --store outputs/history.json
```

Repeat with a later timestamp and photograph. The command reports the change in
visible score per day. It does not convert that change into remaining life.
Item identity, camera view, lighting, and storage metadata are user-supplied.

## Synthetic smoke test

```bash
use-first make-demo --output examples/synthetic
use-first rank \
  --model artifacts/banana_stage_model.joblib \
  --items examples/synthetic/items.csv \
  --output examples/synthetic/ranking.json
```

These generated shapes only exercise the file and ranking pipeline. They are
not real bananas and their outputs are not validation evidence.

## Model pipeline

1. Correct EXIF orientation and resize the crop to at most 384 pixels per side.
2. Extract 130 interpretable RGB/HSV histogram, channel-statistic, color-area,
   edge, and texture features.
3. Augment training crops with controlled dark, bright, and warm-color variants.
4. Compare class-weighted logistic regression and a 300-tree random forest on
   a held-out validation split.
5. Refit the selected model on training plus validation images and evaluate once
   on source-image-isolated test data.
6. Convert class probabilities for stages 0, 1, and 2 into an expected-stage
   visible score. Confidence, normalized entropy, and image-quality checks
   determine whether the result should be reviewed.

The random forest won model selection for this run. The 7.3 MB model artifact is
included. No pretrained neural network is required, which keeps CPU inference
small and fully local. This is an established feature-based baseline suitable
for a manageable prototype, rather than a claim of a new algorithm.

## Dataset and reproduction

Validation used the Apache-2.0 `Fruit-freshness-detection-dataset` at commit
`0cdcf30cb86a6a051b666d9138b5436264972040`. The source contains VOC bounding
boxes for fresh, normal, and rotten examples of several fruit types. This
project extracts banana objects only. Source images are not redistributed.

```bash
git clone https://github.com/zijianchen98/Fruit-freshness-detection-dataset.git data/source
use-first prepare-voc --source data/source --output data/banana --seed 18
use-first audit --manifest data/banana/manifest.csv --output reports/split-audit.json
use-first fit \
  --manifest data/banana/manifest.csv \
  --model artifacts/banana_stage_model.joblib \
  --report reports/validation.json \
  --seed 18
```

All objects from one source photograph remain in the same split, including
photographs with multiple annotated stages. Counts, provenance, and license
notes are in [DATASET.md](DATASET.md).

## Actual validation results

The untouched test set contained 355 banana crops from 218 source photographs.
The splitter grouped 1,806 source photographs into 499 visually similar groups
before partitioning. An audit found no exact byte or decoded-pixel duplicates
across splits and flagged 14 of 355 test crops as conservative visual-similarity
candidates for review.

| Condition | Accuracy | Macro F1 | Ordinal MAE |
| --- | ---: | ---: | ---: |
| Original held-out crops | 100.00% | 100.00% | 0.035 stages |
| Digital brightness × 0.7 | 100.00% | 100.00% | 0.040 stages |
| Digital brightness × 1.3 | 100.00% | 100.00% | 0.042 stages |
| Warm color cast | 100.00% | 100.00% | 0.039 stages |

The test confusion matrix, ordered fresh/normal/rotten, was
`[[84, 0, 0], [0, 211, 0], [0, 0, 60]]`. The augmentation modes match the
stress transformations, so these checks measure controlled robustness rather
than performance under every real camera or lighting shift. The perfect
same-dataset class decision score may reflect acquisition, background,
repeated-subject, or annotation cues.
There was no external-dataset or real longitudinal validation. See
[VALIDATION.md](VALIDATION.md) and the machine-readable
[report](reports/validation.json).

## Tests

```bash
python -m unittest discover -s tests -v
```

Nine tests cover feature stability, quality flags, deterministic splitting,
source-image leakage rejection, bounded storage weighting, uncertain-item
review, duplicate observations, chronological trend calculation, and invalid
storage metadata. The recorded run is in `test-results.txt`.

## Recent primary research and datasets

- Oliveira-Montiel et al. evaluate reproducibility, provenance, efficiency, and
  external validation for banana postharvest models. Their analysis motivates
  the explicit split and provenance reporting here. https://doi.org/10.3390/agriengineering8110459
- Ali et al. introduce BananaImageBD with multiple varieties and four ripeness
  stages. It is an appropriate future external-validation source and was not
  used in this run. https://doi.org/10.1016/j.dib.2024.111239
- Nizamani et al. study RGB-based banana ripeness and quality classification,
  including day-linked acquisition. Their reported system and results were not
  reproduced here. https://doi.org/10.3390/pr13071982
- Lim et al. study weakly supervised defect segmentation for banana grading,
  illustrating the value of localized surface-defect evidence. This prototype
  currently classifies a whole crop. https://arxiv.org/abs/2411.16219

## Limitations

- Three repository labels are treated as ordered visible stages, but the
  dataset does not establish chemical, sensory, microbial, or internal quality.
- Test images come from the same dataset and acquisition process as training.
  Similar-looking photos or shared backgrounds may inflate internal accuracy.
- Bounding-box crops are used for training and evaluation. Busy scenes, several
  overlapping bunches, stickers, bags, unusual cultivars, and partial bananas
  are unvalidated.
- The uncertainty thresholds were operational choices. They routed 2.25% of
  correctly classified original test crops to review. There were no test errors
  from which to estimate error-review recall.
- Digital perturbations do not represent shadows, mixed illumination, blur,
  camera processing, or field-to-shop domain shift.
- Storage days are optional user metadata and act only as a bounded tiebreaker.
  Temperature, humidity, cultivar, handling, and hidden damage are not modeled.
- Longitudinal tracking logic was tested with controlled records, but no real
  tagged-banana time series was evaluated.

Use the output to organize inspection, then apply normal food-handling judgment
and local guidance.

## License

Original software and documentation are released under the [MIT License](LICENSE).
The fitted model is derived from the separately licensed dataset described in
[MODEL_CARD.md](MODEL_CARD.md). Third-party images are not included.

