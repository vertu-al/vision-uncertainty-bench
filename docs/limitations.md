# Limitations and non-claims

This repository is a controlled evaluation benchmark, not evidence that an image classifier is safe for deployment.

## Statistical scope

- The current results come from one trained baseline seed.
- Reported differences should therefore be treated as descriptive baseline findings, not population-level estimates.
- A future version should repeat training across multiple seeds and report uncertainty intervals across runs.

## Dataset scope

- CIFAR-10 is a small, curated benchmark with 32 × 32 pixel images.
- Its classes and visual statistics do not represent real-world deployment environments.
- Clean CIFAR-10 validation and test performance should not be interpreted as evidence of broad visual competence.

## Corruption scope

- Gaussian noise, contrast reduction, and central occlusion are synthetic and deliberately narrow stress tests.
- They do not represent the full diversity of natural distribution shift, sensor failure, lighting changes, motion blur, weather, demographic variation, or dataset shift.
- They are not adversarial attacks.

## Calibration scope

- Temperature scaling learns one scalar temperature from clean validation logits.
- It does not change model weights, predicted classes, or the ranking mechanism used by the classifier.
- Improved calibration metrics do not imply improved robustness or reliable uncertainty estimates under every shift.

## Selective prediction scope

- Confidence-threshold abstention retains predictions only when maximum class probability exceeds a fixed threshold.
- Risk is measured only among retained predictions.
- This is not equivalent to out-of-distribution detection, error detection, conformal prediction, Bayesian uncertainty estimation, or a deployment safety guarantee.
- A threshold that behaves reasonably on clean data may remain unsafe under severe corruption.

## Evaluation scope

- The test set is held out from model selection and temperature fitting, but the present benchmark is still an initial single-project evaluation.
- Expected calibration error depends on binning choices and should be interpreted alongside negative log-likelihood, Brier score, accuracy, coverage, and risk.
- The chosen threshold of 0.80 is one preconfigured operating point, not a universal decision threshold.

## Non-claims

This repository does not claim:

- adversarial robustness;
- causal visual understanding;
- real-world safety;
- clinical, legal, or safety-critical suitability;
- general out-of-distribution detection ability;
- superiority over dedicated uncertainty-estimation methods.

## What this repository does demonstrate

The benchmark demonstrates a reproducible workflow for separating:

1. predictive accuracy,
2. probability calibration,
3. confidence-based abstention, and
4. failure under controlled degradation of visual evidence.

Those distinctions are the central object of study.