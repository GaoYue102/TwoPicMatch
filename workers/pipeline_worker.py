import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import traceback

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.image_io import imread_unicode
from core.params import DetectionParams
from core.feature_matching import compute_feature_match, draw_matches_from_result, validate_homography
from core.alignment import compute_alignment
from core.normalization import normalize_illumination
from core.difference_detector import detect_differences, detect_differences_ssim_only
from core.debug_export import (
    save_original, save_feature_matches, save_alignment,
    save_normalized, save_final_overlay,
)


@dataclass
class PipelineResult:
    ref_path: str = ""
    test_path: str = ""
    ref_bgr: Optional[np.ndarray] = None
    test_bgr: Optional[np.ndarray] = None
    test_warped: Optional[np.ndarray] = None
    ref_norm: Optional[np.ndarray] = None
    common_mask: Optional[np.ndarray] = None
    diff_mask: Optional[np.ndarray] = None
    ssim_map: Optional[np.ndarray] = None    # per-pixel SSIM similarity (1=identical, 0=dissimilar)
    bboxes: List[Tuple[int, int, int, int]] = None
    match_viz: Optional[np.ndarray] = None
    stats: dict = None

    def __post_init__(self):
        if self.bboxes is None:
            self.bboxes = []
        if self.stats is None:
            self.stats = {}


class PipelineWorker(QThread):
    """Runs the full detection pipeline in a background thread."""

    # progress signals
    stage_changed = pyqtSignal(str)          # human-readable stage name
    progress = pyqtSignal(float)             # 0..1
    finished = pyqtSignal(object)            # PipelineResult
    failed = pyqtSignal(str)                 # error message

    def __init__(self, ref_path: str, test_path: str, params: DetectionParams,
                 parent=None):
        super().__init__(parent)
        self._ref_path = ref_path
        self._test_path = test_path
        self._params = params

    def run(self):
        try:
            result = PipelineResult(ref_path=self._ref_path,
                                    test_path=self._test_path,
                                    stats={})
            self._run_pipeline(result)
            self.progress.emit(1.0)
            self.stage_changed.emit("完成")
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    def _run_pipeline(self, r: PipelineResult):
        # Stage 1 — load and validate
        self.stage_changed.emit("加载图像…")
        self.progress.emit(0.05)
        r.ref_bgr = imread_unicode(self._ref_path)
        r.test_bgr = imread_unicode(self._test_path)

        if self._params.debug_export:
            save_original(r.ref_bgr, r.test_bgr)

        from core.image_io import validate_image_pair, images_are_identical

        img_warnings = validate_image_pair(r.ref_bgr, r.test_bgr)
        if img_warnings:
            r.stats["image_warnings"] = img_warnings

        if images_are_identical(r.ref_bgr, r.test_bgr):
            # emit 1 bbox of zero area as a special marker
            r.bboxes = []
            r.diff_mask = np.zeros(r.ref_bgr.shape[:2], dtype=bool)
            r.stats["num_diffs"] = 0
            self.stage_changed.emit("完成 — 两图完全相同")
            self.progress.emit(1.0)
            self.finished.emit(r)
            return

        # Stage 2 — feature matching on downscaled versions
        self.stage_changed.emit("特征点匹配…")
        self.progress.emit(0.1)

        # work on downscaled images for speed
        max_side = self._params.max_matching_side or 2048
        scale = 1.0
        ref_match = r.ref_bgr
        test_match = r.test_bgr
        if max_side > 0 and (
            max(r.ref_bgr.shape[:2]) > max_side
            or max(r.test_bgr.shape[:2]) > max_side
        ):
            scale = max_side / max(*r.ref_bgr.shape[:2], *r.test_bgr.shape[:2])
            new_w = int(r.ref_bgr.shape[1] * scale)
            new_h = int(r.ref_bgr.shape[0] * scale)
            ref_match = cv2.resize(r.ref_bgr, (new_w, new_h))
            test_match = cv2.resize(r.test_bgr, (new_w, new_h))

        fm_result = compute_feature_match(
            ref_match, test_match,
            feature_type=self._params.feature_type,
            ratio_threshold=self._params.feature_threshold,
            ransac_threshold=self._params.ransac_threshold,
            akaze_threshold=self._params.akaze_threshold,
            ransac_confidence=self._params.ransac_confidence,
        )
        H_small = fm_result.H
        r.stats.update(fm_result.stats)
        self.progress.emit(0.3)

        if H_small is not None and scale != 1.0:
            # H_small maps test_small→ref_small; we need H_full: test_full→ref_full
            # small = S * full, so S = diag(scale, scale, 1)
            # H_full = S⁻¹ @ H_small @ S
            S = np.diag([scale, scale, 1.0])
            S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])
            H_full = S_inv @ H_small @ S
        else:
            H_full = H_small

        if H_full is None:
            raise RuntimeError(
                f"特征匹配失败：仅找到 {r.stats.get('num_matches', 0)} 对匹配点。"
                "请尝试更换特征算法或调低 ratio 阈值。"
            )

        h_warnings = validate_homography(
            H_full, r.ref_bgr.shape,
            inlier_ratio=r.stats.get("inlier_ratio", 0),
        )
        if h_warnings:
            r.stats["homography_warnings"] = h_warnings
            critical = any(
                "镜像" in w or "退化" in w for w in h_warnings
            )
            if critical:
                raise RuntimeError(
                    f"单应性可能无效: {'; '.join(h_warnings)}"
                )

        # Stage 3 — alignment
        self.stage_changed.emit("图像对齐…")
        self.progress.emit(0.4)
        r.test_warped, r.common_mask = compute_alignment(
            r.ref_bgr, r.test_bgr, H_full,
            border_margin=self._params.border_margin,
        )
        if r.test_warped is None:
            raise RuntimeError("图像对齐失败。")

        if self._params.debug_export:
            save_alignment(r.test_warped, r.common_mask)

        # Stage 4 — normalization
        self.stage_changed.emit("光照归一化…")
        self.progress.emit(0.5)
        ref_for_detection, r.test_warped = normalize_illumination(
            r.test_warped, r.ref_bgr, r.common_mask,
            do_hist_match=self._params.do_histogram_match,
            clahe_clip=self._params.clahe_clip_limit,
            clahe_tile=self._params.clahe_tile_size,
        )
        r.ref_norm = ref_for_detection
        self.progress.emit(0.6)

        if self._params.debug_export:
            save_normalized(r.ref_norm, r.test_warped)

        # Stage 5 — difference detection
        self.stage_changed.emit("差异检测…")
        self.progress.emit(0.7)
        if self._params.ssim_only_mode:
            bboxes, diff_mask, ssim_map = detect_differences_ssim_only(
                ref_for_detection, r.test_warped, r.common_mask,
                gaussian_blur_sigma=self._params.gaussian_blur_sigma,
                ssim_window=self._params.ssim_window,
                min_area=self._params.min_area,
                use_color_ssim=self._params.use_color_ssim,
                max_detection_side=self._params.max_detection_side,
                close_kernel_size=self._params.morph_close_kernel_size,
                save_debug=self._params.debug_export,
            )
        else:
            bboxes, diff_mask, ssim_map = detect_differences(
                ref_for_detection, r.test_warped, r.common_mask,
                gaussian_blur_sigma=self._params.gaussian_blur_sigma,
                ssim_window=self._params.ssim_window,
                ssim_threshold=self._params.ssim_threshold,
                edge_low=self._params.edge_threshold_low,
                edge_high=self._params.edge_threshold_high,
                edge_density_threshold=self._params.edge_density_threshold,
                fusion_mode=self._params.fusion_mode,
                morph_open_kernel_size=self._params.morph_open_kernel_size,
                morph_close_kernel_size=self._params.morph_close_kernel_size,
                min_area=self._params.min_area,
                use_color_ssim=self._params.use_color_ssim,
                max_detection_side=self._params.max_detection_side,
                min_color_ratio=self._params.min_color_ratio,
                max_bbox_image_ratio=self._params.max_bbox_image_ratio,
                use_colorfulness_mask=self._params.use_colorfulness_mask,
                colorfulness_block_size=self._params.colorfulness_block_size,
                colorfulness_ratio_threshold=self._params.colorfulness_ratio_threshold,
                save_debug=self._params.debug_export,
            )
        r.bboxes = bboxes
        r.diff_mask = diff_mask
        r.ssim_map = ssim_map
        r.stats["num_diffs"] = len(bboxes)
        self.progress.emit(0.95)

        if self._params.debug_export:
            save_final_overlay(r.ref_bgr, r.test_warped, r.bboxes)

        # Stage 6 — match viz (reuses cached fm_result)
        r.match_viz = draw_matches_from_result(
            fm_result,
            ref_match, test_match,
        )

        if self._params.debug_export:
            save_feature_matches(r.match_viz)
