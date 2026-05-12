from typing import Optional, Tuple

import cv2
import numpy as np


def warp_image(
    img: np.ndarray, H: np.ndarray, output_size: Tuple[int, int]
) -> np.ndarray:
    """Perspective-warp img using homography H into a canvas of output_size."""
    return cv2.warpPerspective(img, H, output_size, flags=cv2.INTER_LINEAR)


def common_region_mask(
    ref_shape: Tuple[int, int],
    test_shape: Tuple[int, int],
    H_test_to_ref: np.ndarray,
    border_margin: int = 5,
) -> np.ndarray:
    """Compute a bool mask of the region valid for comparison.

    Excludes:
    - Areas outside the projected test-image quadrilateral (black borders)
    - Areas outside the reference image bounds
    - A margin around the boundary to avoid interpolation bleed.
    """
    h, w = ref_shape[:2]
    th, tw = test_shape[:2]

    # project test-image corners into ref coordinate system
    test_corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    test_corners_ref = cv2.perspectiveTransform(test_corners, H_test_to_ref).reshape(-1, 2)

    # fill the projected quad
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = test_corners_ref.astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)

    # intersect with reference-image bounds
    mask[0:1, :] = 0
    mask[-1:, :] = 0
    mask[:, 0:1] = 0
    mask[:, -1:] = 0

    # erode inward to avoid interpolation artifacts at borders
    if border_margin > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (border_margin * 2 + 1, border_margin * 2 + 1),
        )
        mask = cv2.erode(mask, kernel)

    return mask > 0


def compute_alignment(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    H: Optional[np.ndarray],
    border_margin: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Warp test_img into ref_img coordinates and produce common-region mask.

    Returns (warped_test, common_mask). Both are None if H is None.
    """
    if H is None:
        return None, None

    h, w = ref_img.shape[:2]
    warped = warp_image(test_img, H, (w, h))
    mask = common_region_mask(ref_img.shape, test_img.shape, H, border_margin)
    return warped, mask
