# Methodology

## Research question

Do image classifiers remain appropriately uncertain when visual evidence becomes
progressively less reliable?

## Experimental setting

The benchmark uses CIFAR-10, a ten-class natural-image classification dataset.
A compact convolutional classifier will be trained on clean training images only.

Training, temperature fitting, and final evaluation are deliberately separated:

1. **Training split:** fit model parameters on clean images.
2. **Validation split:** fit the temperature-scaling parameter using clean,
   held-out validation predictions.
3. **Test split:** report final metrics once on clean and corrupted test images.

No corrupted test examples will be used to select model architecture,
hyperparameters, temperature, or abstention threshold.

## Controlled corruptions

All images are floating-point tensors in the interval `[0, 1]`. Each
corruption has three pre-specified ordinal severity levels.

| Corruption | Severity 1 | Severity 2 | Severity 3 |
|---|---:|---:|---:|
| Gaussian noise standard deviation | 0.04 | 0.08 | 0.16 |
| Remaining contrast factor | 0.75 | 0.50 | 0.25 |
| Central black occlusion side length | 6 px | 10 px | 14 px |

Gaussian noise is deterministic for a fixed image-level seed. Contrast
reduction preserves each channel's mean intensity. Central occlusion removes
visual evidence at the image centre.

## Predictive conditions

The benchmark will compare:

1. raw softmax probabilities;
2. post-hoc temperature-scaled probabilities;
3. confidence-threshold abstention, where low-confidence predictions are
   withheld rather than forced.

Temperature scaling is fitted on clean validation data only. It is not assumed
to improve calibration under corruption; this is an empirical question.

## Metrics

For every condition, the benchmark will report:

- accuracy;
- negative log-likelihood (NLL);
- multiclass Brier score;
- expected calibration error (ECE, 15 equal-width confidence bins);
- risk and coverage under confidence-threshold abstention.

The multiclass Brier score is the mean, across images, of the sum of squared
error between the predicted probability vector and the one-hot class target.

## Reproducibility protocol

Experiments will use fixed, recorded seeds. Final claims will be based on
multiple independent training seeds, with both per-seed results and aggregated
summary statistics retained.

## Scope and limitations

This is a controlled benchmark, not a claim of real-world safety. CIFAR-10
corruptions are not clinical images, adversarial attacks, open-world
out-of-distribution detection, or a complete account of deployment risk.