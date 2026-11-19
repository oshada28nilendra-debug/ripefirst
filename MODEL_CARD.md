# Model card

## Intended use

`artifacts/banana_stage_model.joblib` ranks cropped banana photographs by three
visible appearance stages for stock-rotation assistance and research demos.
It is not intended for safety certification, shelf-life prediction, disease
diagnosis, pricing, or automated disposal.

## Model

- Estimator: scikit-learn random forest, 300 trees
- Inputs: one RGB banana crop
- Features: 130 color, distribution, edge, and texture measurements
- Outputs: fresh/normal/rotten probabilities, expected-stage score, review flag
- Artifact size: approximately 7.3 MB
- Training seed: 18

## Training data and license

The model was fitted to banana crops derived from
https://github.com/zijianchen98/Fruit-freshness-detection-dataset at commit
`0cdcf30cb86a6a051b666d9138b5436264972040`. GitHub reports the source
repository license as Apache-2.0. The source images are not included here.

The surrounding original code and documentation are MIT-licensed. Retain this
model card and the dataset attribution when redistributing the fitted artifact.

## Performance and risks

See `VALIDATION.md`. Internal accuracy was 100% on 355 crops after visual-group
splitting. This is same-dataset validation. Users should inspect uncertain
outputs and validate on their own
cultivar, camera, background, and workflow before operational use.

