from typing import Optional

import cv2
import numpy as np


def match_histograms(
    src: np.ndarray, ref: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Match the histogram of src to ref (grayscale or BGR).

    When mask is provided, only pixels inside the mask contribute to the
    histogram model, but the full src is transformed.
    """
    if src.ndim == 2 and ref.ndim == 2:
        # grayscale
        if mask is not None:
            ref_masked = ref[mask]
            src_matched = np.empty_like(src)
            src_matched[mask] = _match_hist_1d(src[mask], ref_masked)
            src_matched[~mask] = src[~mask]
            return src_matched
        return _match_hist_1d(src, ref)

    # multi-channel → match each channel independently
    result = np.empty_like(src)
    for c in range(src.shape[2]):
        src_c = src[:, :, c]
        ref_c = ref[:, :, c]
        if mask is not None:
            result_c = np.empty_like(src_c)
            result_c[mask] = _match_hist_1d(src_c[mask], ref_c[mask])
            result_c[~mask] = src_c[~mask]
            result[:, :, c] = result_c
        else:
            result[:, :, c] = _match_hist_1d(src_c, ref_c)
    return result


def _match_hist_1d(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Histogram matching for a 1-d array / single channel."""
    # CDF of source
    src_vals, src_counts = np.unique(src, return_counts=True)
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]

    # CDF of reference
    ref_vals, ref_counts = np.unique(ref, return_counts=True)
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # map each source value to the reference value with the closest CDF
    interp = np.interp(src_cdf, ref_cdf, ref_vals)
    lut = np.interp(np.arange(256), src_vals, interp).astype(src.dtype)
    return lut[src]


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE local histogram equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if img.ndim == 2:
        return clahe.apply(img)

    # BGR → LAB, apply to L channel only
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def normalize_illumination(
    test_warped: np.ndarray,
    ref_img: np.ndarray,
    common_mask: np.ndarray,
    *,
    do_hist_match: bool,
    clahe_clip: float,
    clahe_tile: int,
) -> tuple:
    """Full illumination normalization pipeline.

    Applies CLAHE to both images for symmetric local contrast, then
    optionally matches the test histogram to ref.

    Returns (normalized_ref, normalized_test).
    """
    ref = apply_clahe(ref_img, clip_limit=clahe_clip, tile_size=clahe_tile)
    result = apply_clahe(test_warped, clip_limit=clahe_clip, tile_size=clahe_tile)

    if do_hist_match:
        result = match_histograms(result, ref, common_mask)

    return ref, result
