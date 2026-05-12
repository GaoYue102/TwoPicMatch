import numpy as np


class TestDifferenceDetector:
    def _call_detect(self, ref, test, mask, params):
        from core.difference_detector import detect_differences
        return detect_differences(
            ref, test, mask,
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

    def test_identical_images_no_diffs(self, identical_pair, default_params):
        ref, test = identical_pair
        mask = np.ones(ref.shape[:2], dtype=bool)
        bboxes, _, _ = self._call_detect(ref, test, mask, default_params)
        assert len(bboxes) == 0

    def test_obvious_diff_detected(self, simple_ref, simple_test, default_params):
        mask = np.ones(simple_ref.shape[:2], dtype=bool)
        bboxes, _, _ = self._call_detect(simple_ref, simple_test, mask, default_params)
        assert len(bboxes) >= 1

    def test_color_ssim_vs_grayscale(self, simple_ref, simple_test, default_params):
        mask = np.ones(simple_ref.shape[:2], dtype=bool)

        default_params.use_color_ssim = True
        bboxes_color, _, _ = self._call_detect(simple_ref, simple_test, mask, default_params)

        default_params.use_color_ssim = False
        bboxes_gray, _, _ = self._call_detect(simple_ref, simple_test, mask, default_params)

        # Both should find differences
        assert len(bboxes_color) >= 1
        assert len(bboxes_gray) >= 1

    def test_downscale_produces_similar_results(self, simple_ref, simple_test, default_params):
        mask = np.ones(simple_ref.shape[:2], dtype=bool)
        default_params.max_detection_side = 0  # no downscale
        bboxes_full, _, _ = self._call_detect(simple_ref, simple_test, mask, default_params)
        default_params.max_detection_side = 128  # heavy downscale
        bboxes_ds, _, _ = self._call_detect(simple_ref, simple_test, mask, default_params)
        # Both should find differences
        assert len(bboxes_full) >= 1
        assert len(bboxes_ds) >= 1
