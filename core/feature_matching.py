from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FeatureMatchResult:
    """Cached result of feature detection + matching + homography."""
    ref_img_shape: Tuple[int, int]
    test_img_shape: Tuple[int, int]
    kp1: list
    kp2: list
    des1: Optional[np.ndarray]
    des2: Optional[np.ndarray]
    good_matches: list
    H: Optional[np.ndarray]
    stats: dict = field(default_factory=dict)


def _create_detector(feature_type: str, akaze_threshold: float = 0.0005):
    """Create feature detector with tuned parameters for dense keypoints."""
    feature_type = feature_type.upper()
    if feature_type == "SIFT":
        return cv2.SIFT_create(
            nfeatures=0,               # keep all detected keypoints
            nOctaveLayers=5,           # more layers → better scale coverage
            contrastThreshold=0.02,    # lower → more keypoints (default 0.04)
            edgeThreshold=5,           # lower → fewer keypoints filtered at edges
            sigma=1.6,
        )
    elif feature_type == "ORB":
        return cv2.ORB_create(nfeatures=8000)
    else:
        return cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,         # full descriptor
            descriptor_channels=3,
            threshold=akaze_threshold, # lower → more keypoints (default 0.001)
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv2.KAZE_DIFF_PM_G2,
        )


def _create_matcher(feature_type: str):
    feature_type = feature_type.upper()
    if feature_type == "ORB":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def compute_feature_match(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    feature_type: str = "AKAZE",
    ratio_threshold: float = 0.7,
    ransac_threshold: float = 5.0,
    akaze_threshold: float = 0.0005,
    ransac_confidence: float = 0.999,
) -> FeatureMatchResult:
    """Compute feature detection, matching, and homography in one pass.

    Returns a FeatureMatchResult that can be reused for match visualization
    without recomputing features.
    """
    detector = _create_detector(feature_type, akaze_threshold)
    matcher = _create_matcher(feature_type)

    kp1, des1 = detector.detectAndCompute(ref_img, None)
    kp2, des2 = detector.detectAndCompute(test_img, None)

    stats = {"num_keypoints_ref": len(kp1), "num_keypoints_test": len(kp2)}
    shape = (ref_img.shape[:2], test_img.shape[:2])

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        stats.update(num_matches=0, num_inliers=0, inlier_ratio=0.0)
        return FeatureMatchResult(
            shape[0], shape[1], kp1, kp2, des1, des2, [], None, stats,
        )

    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < ratio_threshold * n.distance]
    stats["num_matches"] = len(good)

    if len(good) < 4:
        stats.update(num_inliers=0, inlier_ratio=0.0)
        return FeatureMatchResult(
            shape[0], shape[1], kp1, kp2, des1, des2, good, None, stats,
        )

    test_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # USAC_MAGSAC: more accurate than RANSAC, handles outliers better
    H, mask = cv2.findHomography(
        test_pts, ref_pts,
        cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=ransac_confidence,
        maxIters=5000,
    )

    if H is None:
        # Fallback to RANSAC
        H, mask = cv2.findHomography(
            test_pts, ref_pts, cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=ransac_confidence,
            maxIters=5000,
        )

    if H is None:
        stats.update(num_inliers=0, inlier_ratio=0.0)
    else:
        num_inliers = int(mask.sum())
        stats["num_inliers"] = num_inliers
        stats["inlier_ratio"] = num_inliers / len(good) if good else 0.0

    return FeatureMatchResult(
        shape[0], shape[1], kp1, kp2, des1, des2, good, H, stats,
    )


def find_homography(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    feature_type: str = "AKAZE",
    ratio_threshold: float = 0.7,
    ransac_threshold: float = 5.0,
    akaze_threshold: float = 0.0005,
    ransac_confidence: float = 0.999,
) -> Tuple[Optional[np.ndarray], dict]:
    """Compute homography from test_img to ref_img via feature matching.

    Thin wrapper around compute_feature_match() for backward compatibility.
    Returns (H, stats).
    """
    result = compute_feature_match(
        ref_img, test_img, feature_type, ratio_threshold, ransac_threshold,
        akaze_threshold, ransac_confidence,
    )
    return result.H, result.stats


def validate_homography(
    H: np.ndarray,
    img_shape: Tuple[int, int],
    inlier_ratio: float = 0.0,
    min_inlier_ratio: float = 0.1,
    max_scale: float = 3.0,
    max_shear: float = 0.5,
) -> list:
    """Validate homography quality. Returns list of warning messages.

    Checks: inlier ratio, extreme scale/shear, degenerate/reflected matrix,
    and whether corners project within reasonable bounds.
    """
    warnings = []

    if inlier_ratio < min_inlier_ratio:
        warnings.append(f"内点比例过低: {inlier_ratio:.1%}")

    scale_x = abs(H[0, 0])
    scale_y = abs(H[1, 1])
    if scale_x > max_scale or scale_x < 1.0 / max_scale:
        warnings.append(f"X方向缩放异常: {scale_x:.2f}")
    if scale_y > max_scale or scale_y < 1.0 / max_scale:
        warnings.append(f"Y方向缩放异常: {scale_y:.2f}")

    shear_x = abs(H[0, 1]) / max(abs(H[0, 0]), 1e-6)
    shear_y = abs(H[1, 0]) / max(abs(H[1, 1]), 1e-6)
    if shear_x > max_shear:
        warnings.append(f"X方向剪切异常: {shear_x:.2f}")
    if shear_y > max_shear:
        warnings.append(f"Y方向剪切异常: {shear_y:.2f}")

    det = np.linalg.det(H[:2, :2])
    if det <= 0:
        warnings.append(f"镜像/反射单应性 (det={det:.4f})")
    elif det < 0.01:
        warnings.append(f"近似退化单应性 (det={det:.4f})")

    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if np.any(projected < -w * 2) or np.any(projected > w * 3):
        warnings.append("单应性投影超出图像边界")

    return warnings


def draw_matches_from_result(
    result: FeatureMatchResult,
    ref_img: np.ndarray,
    test_img: np.ndarray,
    max_draw: int = 100,
) -> Optional[np.ndarray]:
    """Draw side-by-side match viz from a pre-computed FeatureMatchResult."""
    if result.des1 is None or result.des2 is None:
        return None
    good = result.good_matches[:max_draw]
    return cv2.drawMatches(
        ref_img, result.kp1, test_img, result.kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_matches(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    feature_type: str = "AKAZE",
    ratio_threshold: float = 0.7,
    max_draw: int = 100,
) -> Optional[np.ndarray]:
    """Return a side-by-side match-visualization image (for debugging).

    Thin wrapper that recomputes features. Prefer draw_matches_from_result()
    when a FeatureMatchResult is already available.
    """
    result = compute_feature_match(
        ref_img, test_img, feature_type, ratio_threshold,
    )
    return draw_matches_from_result(result, ref_img, test_img, max_draw)
