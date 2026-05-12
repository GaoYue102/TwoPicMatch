from dataclasses import dataclass


@dataclass
class DetectionParams:
    # --- feature matching ---
    feature_type: str = "AKAZE"
    feature_threshold: float = 0.7
    ransac_threshold: float = 5.0
    max_matching_side: int = 2048           # downsample for feature matching (0=no limit)
    akaze_threshold: float = 0.0005         # AKAZE detection threshold (lower→more keypoints)
    ransac_confidence: float = 0.999        # RANSAC confidence (higher→more accurate H)

    # --- illumination normalization ---
    do_histogram_match: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # --- pre-processing ---
    gaussian_blur_sigma: float = 1.5          # pre-blur to suppress minor misalignment
    border_margin: int = 5                     # px to shrink common region from edges

    # --- difference detection ---
    ssim_window: int = 48                      # SSIM block size (balance localization vs tolerance)
    ssim_threshold: float = 0.88               # SSIM below this → candidate
    use_color_ssim: bool = True                # per-channel SSIM for color-aware detection
    max_detection_side: int = 2048             # downsample large images for speed (0=no limit)
    min_color_ratio: float = 1.5               # color-diff ratio threshold for bbox filtering
    edge_threshold_low: int = 30               # Canny low threshold
    edge_threshold_high: int = 100             # Canny high threshold
    edge_density_threshold: float = 0.003      # edge-density difference threshold [0–1]
    fusion_mode: str = "AND"                   # "AND" | "OR"

    # --- post-processing ---
    morph_open_kernel_size: int = 5            # opening kernel (removes small noise)
    morph_close_kernel_size: int = 7           # closing kernel (fills holes)
    min_area: int = 500                        # minimum diff region area (px²)
    max_bbox_image_ratio: float = 0.5          # filter bboxes covering >50% of image

    # --- colorfulness pre-mask ---
    use_colorfulness_mask: bool = True           # block-level color-ratio pre-filter
    colorfulness_block_size: int = 32            # analysis block size (px)
    colorfulness_ratio_threshold: float = 2.0    # channel imbalance ratio threshold

    # --- visualization ---
    show_heatmap: bool = False                   # overlay SSIM dissimilarity heatmap on images
    heatmap_opacity: float = 0.7                 # heatmap overlay opacity (0.1–1.0)
    ssim_only_mode: bool = False                 # use SSIM-only detection (skip edge/fusion/morph/color)

    # --- debug ---
    debug_export: bool = False                   # save intermediate step images to debug_steps/

    def to_dict(self) -> dict:
        return {
            "feature_type": self.feature_type,
            "feature_threshold": self.feature_threshold,
            "ransac_threshold": self.ransac_threshold,
            "max_matching_side": self.max_matching_side,
            "akaze_threshold": self.akaze_threshold,
            "ransac_confidence": self.ransac_confidence,
            "do_histogram_match": self.do_histogram_match,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_size": self.clahe_tile_size,
            "gaussian_blur_sigma": self.gaussian_blur_sigma,
            "border_margin": self.border_margin,
            "ssim_window": self.ssim_window,
            "ssim_threshold": self.ssim_threshold,
            "use_color_ssim": self.use_color_ssim,
            "max_detection_side": self.max_detection_side,
            "min_color_ratio": self.min_color_ratio,
            "edge_threshold_low": self.edge_threshold_low,
            "edge_threshold_high": self.edge_threshold_high,
            "edge_density_threshold": self.edge_density_threshold,
            "fusion_mode": self.fusion_mode,
            "morph_open_kernel_size": self.morph_open_kernel_size,
            "morph_close_kernel_size": self.morph_close_kernel_size,
            "min_area": self.min_area,
            "max_bbox_image_ratio": self.max_bbox_image_ratio,
            "use_colorfulness_mask": self.use_colorfulness_mask,
            "colorfulness_block_size": self.colorfulness_block_size,
            "colorfulness_ratio_threshold": self.colorfulness_ratio_threshold,
            "show_heatmap": self.show_heatmap,
            "heatmap_opacity": self.heatmap_opacity,
            "ssim_only_mode": self.ssim_only_mode,
            "debug_export": self.debug_export,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DetectionParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


PRESETS: dict = {
    "默认": {},
    "严格 (高精度)": {
        "ssim_threshold": 0.92, "edge_density_threshold": 0.001,
        "fusion_mode": "AND", "gaussian_blur_sigma": 1.0,
        "morph_open_kernel_size": 3, "min_area": 200,
    },
    "宽松 (高召回)": {
        "ssim_threshold": 0.80, "edge_density_threshold": 0.008,
        "fusion_mode": "OR", "gaussian_blur_sigma": 2.5,
        "morph_open_kernel_size": 7, "min_area": 1000,
    },
    "高速模式": {
        "max_detection_side": 1024, "max_matching_side": 1024,
        "feature_type": "ORB",
    },
}


def params_with_preset(base: "DetectionParams", preset_name: str) -> "DetectionParams":
    """Apply a named preset on top of base params."""
    overrides = PRESETS.get(preset_name, {})
    d = base.to_dict()
    d.update(overrides)
    return DetectionParams.from_dict(d)
