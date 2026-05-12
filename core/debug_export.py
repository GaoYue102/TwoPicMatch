"""Debug image export — saves key pipeline step images for human inspection.

All images are saved to a fixed set of filenames in debug_steps/,
overwritten on each run so the user can review the latest processing.
"""

import os
import cv2
import numpy as np

_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_steps")


def _ensure_dir():
    os.makedirs(_OUT_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(_OUT_DIR, name)


def _save(name: str, img: np.ndarray):
    """Save image, converting bool/float to uint8 as needed."""
    _ensure_dir()
    path = _path(name)
    if img.dtype == bool:
        img = (img.astype(np.uint8) * 255)
    elif img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def _hstack_ensure_same_height(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack two images horizontally, padding the shorter one."""
    if a.shape[0] != b.shape[0]:
        max_h = max(a.shape[0], b.shape[0])
        a2 = np.zeros((max_h, a.shape[1], 3), dtype=a.dtype)
        b2 = np.zeros((max_h, b.shape[1], 3), dtype=b.dtype)
        a2[:a.shape[0]] = a
        b2[:b.shape[0]] = b
        return np.hstack([a2, b2])
    return np.hstack([a, b])


def save_original(ref: np.ndarray, test: np.ndarray):
    """Step 1 — original images side-by-side."""
    _save("01_original.png", _hstack_ensure_same_height(ref, test))


def save_feature_matches(match_viz: np.ndarray):
    """Step 2 — feature matching visualization."""
    _save("02_feature_matches.png", match_viz)


def save_alignment(test_warped: np.ndarray, common_mask: np.ndarray):
    """Step 3 — warped test image with common region mask overlay."""
    warped_vis = test_warped.copy()
    mask_3ch = np.stack([common_mask] * 3, axis=-1).astype(np.uint8)
    overlay = (warped_vis.astype(np.float32) * 0.5 + mask_3ch * 128 * 0.5).astype(np.uint8)
    _save("03_alignment_warped.png", overlay)
    _save("03_alignment_mask.png", common_mask)


def save_normalized(ref_norm: np.ndarray, test_norm: np.ndarray):
    """Step 4 — normalized ref and test side-by-side."""
    _save("04_normalized.png", _hstack_ensure_same_height(ref_norm, test_norm))


def save_final_overlay(
    ref: np.ndarray,
    test_warped: np.ndarray,
    bboxes: list,
    contours: list = None,
):
    """Step 10 — final detection overlay on ref and warped test."""
    h, w = ref.shape[:2]
    result = _hstack_ensure_same_height(ref, test_warped)
    num = len(bboxes)

    if contours:
        for i, cnt in enumerate(contours):
            color = (0, 255, 255) if i == 0 else (0, 0, 255)
            cv2.drawContours(result, [cnt], -1, color, 4)
            x, y = cnt[:, 0, :].min(axis=0)
            cv2.putText(result, f"D{i}", (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            offset = np.array([w, 0], dtype=np.int32)
            cv2.drawContours(result, [cnt + offset], -1, color, 4)
            cv2.putText(result, f"D{i}", (int(x) + w + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    else:
        for i, (bx, by, bw, bh) in enumerate(bboxes):
            color = (0, 255, 255) if i == 0 else (0, 0, 255)
            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), color, 4)
            cv2.putText(result, f"D{i}", (bx + 5, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            if test_warped.shape[0] >= by + bh:
                cv2.rectangle(result, (bx + w, by), (bx + bw + w, by + bh), color, 4)
                cv2.putText(result, f"D{i}", (bx + w + 5, by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Legend bar
    band_h = 60
    legend = np.zeros((band_h, result.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, f"Detections: {num}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    result = np.vstack([legend, result])

    if result.shape[1] > 3000:
        scale = 3000 / result.shape[1]
        result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _save("10_final_detections.png", result)
