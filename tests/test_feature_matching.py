import numpy as np


class TestFeatureMatching:
    def test_compute_feature_match_identical(self, identical_pair):
        from core.feature_matching import compute_feature_match
        ref, test = identical_pair
        result = compute_feature_match(ref, test)
        assert result.H is not None
        assert np.allclose(result.H, np.eye(3), atol=0.05)

    def test_compute_feature_match_translated(self):
        from core.feature_matching import compute_feature_match
        # Use a textured image that AKAZE can find keypoints on
        import cv2
        rng = np.random.RandomState(42)
        ref = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        ref = cv2.GaussianBlur(ref, (3, 3), 0)
        M = np.float32([[1, 0, 5], [0, 1, 5]])
        test = cv2.warpAffine(ref, M, (256, 256))
        result = compute_feature_match(ref, test)
        assert result.H is not None
        assert result.stats["num_inliers"] >= 4


    def test_find_homography_delegates_to_compute(self, shifted_pair):
        from core.feature_matching import find_homography, compute_feature_match
        ref, test = shifted_pair
        H1, _ = find_homography(ref, test)
        fm = compute_feature_match(ref, test)
        H2 = fm.H
        if H1 is not None:
            assert H2 is not None
            assert np.allclose(H1, H2, atol=1e-4)

    def test_validate_homography_identity_passes(self):
        from core.feature_matching import validate_homography
        H = np.eye(3, dtype=np.float64)
        warnings = validate_homography(H, (256, 256), inlier_ratio=0.9)
        assert len(warnings) == 0

    def test_validate_homography_low_inliers(self):
        from core.feature_matching import validate_homography
        H = np.eye(3, dtype=np.float64)
        warnings = validate_homography(H, (256, 256), inlier_ratio=0.05)
        assert any("内点比例过低" in w for w in warnings)

    def test_validate_homography_mirrored(self):
        from core.feature_matching import validate_homography
        H = np.eye(3, dtype=np.float64)
        H[0, 0] = -1.0
        warnings = validate_homography(H, (256, 256))
        assert any("镜像" in w for w in warnings)

    def test_draw_matches_from_result(self, shifted_pair):
        from core.feature_matching import compute_feature_match, draw_matches_from_result
        ref, test = shifted_pair
        fm = compute_feature_match(ref, test)
        viz = draw_matches_from_result(fm, ref, test)
        assert viz is not None
        assert viz.shape[0] == ref.shape[0]
        assert viz.shape[1] == ref.shape[1] * 2
