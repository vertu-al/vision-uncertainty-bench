"""Controlled image corruptions for robustness and uncertainty experiments."""

import torch

_NOISE_STANDARD_DEVIATIONS = {1: 0.04, 2: 0.08, 3: 0.16}
_CONTRAST_FACTORS = {1: 0.75, 2: 0.50, 3: 0.25}
_OCCLUSION_SIDE_LENGTHS = {1: 6, 2: 10, 3: 14}


def _validate_image(image: torch.Tensor) -> None:
    if image.ndim != 3:
        raise ValueError("image must have shape [channels, height, width].")

    if not torch.is_floating_point(image):
        raise ValueError("image must use a floating-point dtype.")

    if not torch.isfinite(image).all():
        raise ValueError("image must contain only finite values.")

    if (image < 0.0).any() or (image > 1.0).any():
        raise ValueError("image values must lie in the interval [0, 1].")


def _validate_severity(severity: int) -> None:
    if severity not in _NOISE_STANDARD_DEVIATIONS:
        raise ValueError("severity must be one of 1, 2, or 3.")


def _gaussian_noise(
    image: torch.Tensor,
    severity: int,
    seed: int | None,
) -> torch.Tensor:
    standard_deviation = _NOISE_STANDARD_DEVIATIONS[severity]

    if seed is None:
        noise = torch.randn(image.shape, dtype=image.dtype, device=image.device)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn(
            image.shape,
            dtype=image.dtype,
            device="cpu",
            generator=generator,
        ).to(image.device)

    return (image + standard_deviation * noise).clamp(0.0, 1.0)


def _reduce_contrast(image: torch.Tensor, severity: int) -> torch.Tensor:
    factor = _CONTRAST_FACTORS[severity]
    channel_means = image.mean(dim=(-2, -1), keepdim=True)

    return channel_means + factor * (image - channel_means)


def _apply_central_occlusion(image: torch.Tensor, severity: int) -> torch.Tensor:
    side_length = _OCCLUSION_SIDE_LENGTHS[severity]
    _, height, width = image.shape

    if side_length > height or side_length > width:
        raise ValueError("occlusion is larger than the image dimensions.")

    row_start = (height - side_length) // 2
    column_start = (width - side_length) // 2

    occluded = image.clone()
    occluded[
        :,
        row_start : row_start + side_length,
        column_start : column_start + side_length,
    ] = 0.0
    return occluded


def apply_corruption(
    image: torch.Tensor,
    corruption: str,
    severity: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply one named, controlled corruption to an image in [0, 1].

    Severity is ordinal: 1 is mild, 2 is moderate, and 3 is severe.
    """
    _validate_image(image)
    _validate_severity(severity)

    if corruption == "gaussian_noise":
        return _gaussian_noise(image, severity, seed)

    if corruption == "contrast":
        return _reduce_contrast(image, severity)

    if corruption == "central_occlusion":
        return _apply_central_occlusion(image, severity)

    raise ValueError(f"Unknown corruption: {corruption}")