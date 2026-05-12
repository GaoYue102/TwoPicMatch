import cv2
from pathlib import Path

TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"


class TestIntegration:
    def test_pipeline_simple(self):
        """End-to-end pipeline test with simple test images."""
        from core.params import DetectionParams
        from core.feature_matching import compute_feature_match
        from core.alignment import compute_alignment
        from core.normalization import normalize_illumination
        from core.difference_detector import detect_differences

        ref = cv2.imread(str(TEST_IMAGES_DIR / "ref_simple.png"))
        test = cv2.imread(str(TEST_IMAGES_DIR / "test_simple_diff.png"))
        assert ref is not None and test is not None

        params = DetectionParams()
        fm = compute_feature_match(
            ref, test, params.feature_type,
            params.feature_threshold, params.ransac_threshold,
        )
        assert fm.H is not None, f"Homography failed: {fm.stats}"

        warped, mask = compute_alignment(ref, test, fm.H, params.border_margin)
        assert warped is not None

        ref_norm, test_norm = normalize_illumination(
            warped, ref, mask,
            do_hist_match=params.do_histogram_match,
            clahe_clip=params.clahe_clip_limit,
            clahe_tile=params.clahe_tile_size,
        )

        bboxes, _, _ = detect_differences(
            ref_norm, test_norm, mask,
            gaussian_blur_sigma=params.gaussian_blur_sigma,
            ssim_window=params.ssim_window,
            ssim_threshold=params.ssim_threshold,
            edge_low=params.edge_threshold_low,
            edge_high=params.edge_threshold_high,
            edge_density_threshold=params.edge_density_threshold,
            fusion_mode=params.fusion_mode,
            morph_open_kernel_size=params.morph_open_kernel_size,
            morph_close_kernel_size=params.morph_close_kernel_size,
            min_area=params.min_area,
            use_color_ssim=params.use_color_ssim,
            max_detection_side=params.max_detection_side,
            min_color_ratio=params.min_color_ratio,
        )
        assert len(bboxes) > 0, "Should find differences in test images"

    def test_pipeline_assembly(self):
        """End-to-end pipeline test with assembly test images."""
        from core.params import DetectionParams
        from core.feature_matching import compute_feature_match
        from core.alignment import compute_alignment
        from core.normalization import normalize_illumination
        from core.difference_detector import detect_differences

        ref = cv2.imread(str(TEST_IMAGES_DIR / "ref_assembly.png"))
        test = cv2.imread(str(TEST_IMAGES_DIR / "test_missing_parts.png"))
        assert ref is not None and test is not None

        params = DetectionParams()
        fm = compute_feature_match(
            ref, test, params.feature_type,
            params.feature_threshold, params.ransac_threshold,
        )
        assert fm.H is not None, f"Homography failed: {fm.stats}"

        warped, mask = compute_alignment(ref, test, fm.H, params.border_margin)
        ref_norm, test_norm = normalize_illumination(
            warped, ref, mask,
            do_hist_match=params.do_histogram_match,
            clahe_clip=params.clahe_clip_limit,
            clahe_tile=params.clahe_tile_size,
        )

        bboxes, _, _ = detect_differences(
            ref_norm, test_norm, mask,
            gaussian_blur_sigma=params.gaussian_blur_sigma,
            ssim_window=params.ssim_window,
            ssim_threshold=params.ssim_threshold,
            edge_low=params.edge_threshold_low,
            edge_high=params.edge_threshold_high,
            edge_density_threshold=params.edge_density_threshold,
            fusion_mode=params.fusion_mode,
            morph_open_kernel_size=params.morph_open_kernel_size,
            morph_close_kernel_size=params.morph_close_kernel_size,
            min_area=params.min_area,
            use_color_ssim=params.use_color_ssim,
            max_detection_side=params.max_detection_side,
            min_color_ratio=params.min_color_ratio,
        )
        assert len(bboxes) > 0, "Should find differences in test images"
