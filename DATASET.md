# Dataset record

## Training and evaluation source

- Name: Fruit-freshness-detection-dataset
- Repository: https://github.com/zijianchen98/Fruit-freshness-detection-dataset
- Commit used: `0cdcf30cb86a6a051b666d9138b5436264972040`
- Repository license metadata: Apache License 2.0
- Retrieved: 18 November 2026
- Format: JPEG images and Pascal VOC XML object annotations

The source includes apple, banana, and orange objects. Only objects labeled
`fresh_banana`, `normal_banana`, or `rotten_banana` are used. Each annotated
banana bounding box is cropped into one training unit. Images remain in their
source repository and are excluded from this package.

## Prepared counts

| Split | Fresh | Normal | Rotten | Total crops |
| --- | ---: | ---: | ---: | ---: |
| Train | 582 | 977 | 400 | 1,959 |
| Validation | 238 | 168 | 321 | 727 |
| Test | 84 | 211 | 60 | 355 |
| Total | 904 | 1,356 | 781 | 3,041 |

The 1,806 source photographs are clustered into 499 visual groups using banana
crop difference hashes plus a conservative resized-pixel distance. A
deterministic hash assigns each group to one split, and all objects from every
grouped photograph stay together. The test split contains 218 source
photographs. The grouping creates uneven class counts, which the models address
with class weighting.

## Label meaning

The three labels are used as an ordinal appearance scale:

- 0 `fresh`
- 1 `normal`
- 2 `rotten`

These names are inherited from the source annotations. They are not laboratory
measurements and must not be interpreted as microbial safety or exact remaining
life. The source does not supply verified longitudinal identity, storage
conditions, cultivar, chemical measurements, or sensory scores.

## Reproduction

The `prepare-voc` command validates expected directories, reads VOC boxes,
clips boxes to image dimensions, rejects invalid boxes, preserves source-image
grouping, writes JPEG crops, and emits `manifest.csv` plus
`dataset_summary.json`. The generated `data/` directory is ignored by Git.

