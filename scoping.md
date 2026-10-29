# When Vision Gets It Wrong

## Research question

How do predictive accuracy, confidence calibration, and selective risk change when a clean-image classifier encounters controlled degradation of its visual evidence?

## Motivation

A classifier can preserve apparently high confidence even when blur, noise, contrast loss, or occlusion make its input less reliable. Accuracy alone does not reveal whether a model knows when it may be wrong.

## Hypotheses

1. Accuracy and calibration will degrade as corruption severity increases.
2. Temperature scaling fitted on clean validation data may improve in-distribution calibration; this benchmark tests whether that improvement transfers to corrupted inputs.
3. Confidence-based abstention should lower selective risk at the cost of reduced coverage, but the trade-off may differ by corruption type.

## Experimental design

- Dataset: CIFAR-10.
- Data split: training, held-out validation, untouched test set.
- Reference model: compact convolutional classifier trained only on clean training images.
- Corruptions: Gaussian noise, Gaussian blur, contrast reduction, and central occlusion.
- Severity: three deterministic levels per corruption.
- Probability methods: raw softmax probabilities and post-hoc temperature-scaled probabilities.
- Decision policy: confidence-threshold abstention.

## Metrics

- Accuracy
- Negative log-likelihood
- Brier score
- Expected calibration error
- Reliability diagrams
- Risk–coverage curves
- Per-corruption failure analysis

## Comparison rules

- Every method sees identical test images and corruption instances.
- Temperature is fitted only on validation data.
- The untouched test set is never used for model selection or calibration.
- Results are reported across multiple training seeds.

## Intended outputs

1. Corruption-severity curves for accuracy, NLL, Brier score, and ECE.
2. Reliability diagrams for clean and corrupted conditions.
3. Risk–coverage curves for raw and calibrated confidence.
4. A concise failure-analysis table.

## Non-claims

This is a controlled benchmark, not evidence of safety in real-world vision systems. It does not establish robustness to adversarial attacks, clinical validity, or reliable out-of-distribution detection beyond the specified corruptions.