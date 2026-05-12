from typing import List, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _preprocess(
    ref_gray: np.ndarray,
    test_gray: np.ndarray,
    blur_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian blur to suppress minor misalignment artifacts."""
    if blur_sigma <= 0:
        return ref_gray, test_gray
    ksize = int(blur_sigma * 4 + 1) | 1  # odd kernel size
    ref = cv2.GaussianBlur(ref_gray, (ksize, ksize), blur_sigma)
    tst = cv2.GaussianBlur(test_gray, (ksize, ksize), blur_sigma)
    return ref, tst


def _compute_ssim_map(
    ref: np.ndarray, test: np.ndarray, win_size: int
) -> np.ndarray:
    """Return a per-pixel SSIM similarity map (1 = identical, 0 = dissimilar)."""
    if win_size % 2 == 0:
        win_size += 1
    _, ssim_map = ssim(ref, test, full=True, win_size=win_size, data_range=255)
    return ssim_map


def _compute_ssim_map_color(
    ref_bgr: np.ndarray, test_bgr: np.ndarray, win_size: int
) -> np.ndarray:
    """Per-channel SSIM, returning min(channels) for color-aware comparison.

    Yellow↔gray changes are invisible to grayscale but cause large B-channel
    differences. Taking min across channels makes those visible.
    """
    if win_size % 2 == 0:
        win_size += 1
    ssim_maps = []
    for c in range(3):
        _, sm = ssim(ref_bgr[:, :, c], test_bgr[:, :, c],
                      full=True, win_size=win_size, data_range=255)
        ssim_maps.append(sm)
    return np.minimum(np.minimum(ssim_maps[0], ssim_maps[1]), ssim_maps[2])


def _ssim_difference_map(
    ref_gray: np.ndarray, test_gray: np.ndarray,
    win_size: int, threshold: float,
) -> np.ndarray:
    ssim_map = _compute_ssim_map(ref_gray, test_gray, win_size)
    return ssim_map < threshold


def _edge_density_difference_map(
    ref_gray: np.ndarray, test_gray: np.ndarray,
    low: int, high: int, block_size: int, threshold: float,
) -> np.ndarray:
    edges_ref = cv2.Canny(ref_gray, low, high) / 255.0
    edges_test = cv2.Canny(test_gray, low, high) / 255.0

    dens_ref = cv2.boxFilter(
        edges_ref.astype(np.float32), -1,
        (block_size, block_size), normalize=True,
    )
    dens_test = cv2.boxFilter(
        edges_test.astype(np.float32), -1,
        (block_size, block_size), normalize=True,
    )

    return np.abs(dens_ref - dens_test) > threshold


def _fuse_maps(ssim_mask: np.ndarray, edge_mask: np.ndarray, mode: str) -> np.ndarray:
    return ssim_mask & edge_mask if mode == "AND" else ssim_mask | edge_mask


def _morph_open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def _morph_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def _filter_by_color_diff(
    ref_bgr: np.ndarray,
    test_bgr: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    min_color_ratio: float = 1.5,
) -> List[Tuple[int, int, int, int]]:
    """Filter bboxes: keep only those where the diff is color-significant.

    Lighting/shadow changes affect all channels similarly (gray diff).
    Real object changes (e.g. orange earplug missing) show channel imbalance.
    Ratio = max_channel_diff / min_channel_diff; high ratio = colorful change.
    """
    if not bboxes:
        return bboxes

    ref_f = ref_bgr.astype(np.float32)
    test_f = test_bgr.astype(np.float32)

    kept = []
    for (x, y, w, h) in bboxes:
        y2 = min(y + h, ref_bgr.shape[0])
        x2 = min(x + w, ref_bgr.shape[1])
        patch_ref = ref_f[y:y2, x:x2]
        patch_test = test_f[y:y2, x:x2]

        # per-channel mean absolute difference
        ch_diffs = []
        for c in range(3):
            ch_diffs.append(float(np.abs(patch_ref[:, :, c] - patch_test[:, :, c]).mean()))
        ch_diffs = np.array(ch_diffs)

        # ratio of largest to smallest channel diff
        max_ch = ch_diffs.max()
        min_ch = ch_diffs.min()
        if min_ch < 1.0:
            min_ch = 1.0
        ratio = max_ch / min_ch

        if ratio >= min_color_ratio:
            kept.append((x, y, w, h))

    return kept


def _save_debug(name: str, img: np.ndarray):
    """Save a debug intermediate image to project debug_steps/ directory."""
    import os
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "debug_steps",
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if img.dtype == bool:
        img = (img.astype(np.uint8) * 255)
    elif img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def _compute_colorfulness_mask(
    ref_bgr: np.ndarray,
    test_bgr: np.ndarray,
    block_size: int,
    ratio_threshold: float,
    diff_threshold: float,
) -> np.ndarray:
    """Block-level colorfulness mask to suppress alignment artifacts.

    Alignment noise affects all BGR channels similarly (ratio near 1.0).
    Real object changes cause channel imbalance (ratio >> 1.0).
    Returns a boolean mask where blocks pass both ratio and diff thresholds.
    """
    h, w = ref_bgr.shape[:2]
    ref_f = ref_bgr.astype(np.float32)
    test_f = test_bgr.astype(np.float32)
    bs = max(block_size, 8)
    gh, gw = h // bs, w // bs

    block_mask = np.zeros((gh, gw), dtype=np.uint8)
    for by in range(gh):
        for bx in range(gw):
            y1, y2 = by * bs, min((by + 1) * bs, h)
            x1, x2 = bx * bs, min((bx + 1) * bs, w)
            pr = ref_f[y1:y2, x1:x2]
            pt = test_f[y1:y2, x1:x2]
            ch_means = [float(np.abs(pr[..., c] - pt[..., c]).mean()) for c in range(3)]
            max_ch = max(ch_means)
            min_ch = max(min(ch_means), 0.5)
            ratio = max_ch / min_ch
            mean_diff = float(np.mean(ch_means))
            if ratio > ratio_threshold and mean_diff > diff_threshold:
                block_mask[by, bx] = 1

    if gh == 0 or gw == 0:
        return np.ones((h, w), dtype=bool)

    full_mask = cv2.resize(
        block_mask, (w, h), interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return full_mask


def _extract_bboxes(
    mask: np.ndarray, min_area: int
) -> List[Tuple[int, int, int, int]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    bboxes = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bboxes.append((x, y, w, h))
    return bboxes


def detect_differences(
    ref_bgr: np.ndarray,
    test_warped_bgr: np.ndarray,
    common_mask: np.ndarray,
    *,
    gaussian_blur_sigma: float,
    ssim_window: int,
    ssim_threshold: float,
    edge_low: int,
    edge_high: int,
    edge_density_threshold: float,
    fusion_mode: str,
    morph_open_kernel_size: int,
    morph_close_kernel_size: int,
    min_area: int,
    use_color_ssim: bool,
    max_detection_side: int,
    min_color_ratio: float,
    max_bbox_image_ratio: float = 0.5,
    # --- colorfulness pre-mask (suppress alignment artifacts via channel imbalance) ---
    use_colorfulness_mask: bool = False,
    colorfulness_block_size: int = 32,
    colorfulness_ratio_threshold: float = 2.0,
    # --- debug export ---
    save_debug: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    """Detect structural differences between aligned images.

    Designed for assembly inspection: detects missing / wrong / misoriented
    parts while ignoring minor viewpoint / lighting residuals.

    When use_colorfulness_mask=True, a block-level color-ratio pre-mask
    suppresses alignment artifacts (which affect all channels similarly).
    Useful for handheld/parallax-heavy image pairs.

    If save_debug=True, intermediate maps (SSIM, edge, fused, etc.)
    are saved to debug_steps/ for human inspection.

    Returns (list_of_bboxes, diff_mask_full, ssim_map_0_1) where
    ssim_map_0_1 is the per-pixel SSIM similarity map (1=identical, 0=dissimilar)
    at original image resolution, suitable for heatmap overlay.
    """
    ref_gray = _gray(ref_bgr)
    test_gray = _gray(test_warped_bgr)

    # Step 0 — downscale large images for performance
    ds_scale = 1.0
    _orig_h, _orig_w = ref_bgr.shape[:2]  # saved for exact scale-back
    if max_detection_side > 0:
        max_side = max(ref_bgr.shape[0], ref_bgr.shape[1])
        if max_side > max_detection_side:
            ds_scale = max_detection_side / max_side
            new_w = int(ref_bgr.shape[1] * ds_scale)
            new_h = int(ref_bgr.shape[0] * ds_scale)
            ref_bgr = cv2.resize(ref_bgr, (new_w, new_h))
            test_warped_bgr = cv2.resize(test_warped_bgr, (new_w, new_h))
            common_mask = cv2.resize(
                common_mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            ref_gray = _gray(ref_bgr)
            test_gray = _gray(test_warped_bgr)

    # Step 1 — blur to suppress sub-pixel misalignment noise
    if use_color_ssim:
        ref_blur_bgr, test_blur_bgr = _preprocess(
            ref_bgr, test_warped_bgr, gaussian_blur_sigma
        )
        ref_blur = _gray(ref_blur_bgr)
        test_blur = _gray(test_blur_bgr)
        _ssim_fn = _compute_ssim_map_color
        _ref_for_ssim = ref_blur_bgr
        _test_for_ssim = test_blur_bgr
    else:
        ref_blur, test_blur = _preprocess(ref_gray, test_gray, gaussian_blur_sigma)
        _ssim_fn = _compute_ssim_map
        _ref_for_ssim = ref_blur
        _test_for_ssim = test_blur

    # 用参考图填充检测图的非公共区域，从根源消除 SSIM 窗口跨越边界
    # 产生的伪影（黑边区域的像素差异通过滑动窗口污染边界内侧的 SSIM 值）
    _test_for_ssim = _test_for_ssim.copy()
    _test_for_ssim[~common_mask] = _ref_for_ssim[~common_mask]
    ssim_map = _ssim_fn(_ref_for_ssim, _test_for_ssim, ssim_window)
    ssim_mask = ssim_map < ssim_threshold

    if save_debug:
        _save_debug( "05_ssim_map.png",
                    ((1.0 - np.clip(ssim_map, 0, 1)) * 255).astype(np.uint8))
        # 热力图叠加：JET色映射 + 70%透明度，仅覆盖公共区域
        ssim_dissim = ((1.0 - np.clip(ssim_map, 0, 1)) * 255).astype(np.uint8)
        ssim_heatmap = cv2.applyColorMap(ssim_dissim, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(ssim_heatmap, 0.7, ref_bgr, 0.3, 0)
        overlay = ref_bgr.copy()
        overlay[common_mask] = blended[common_mask]
        _save_debug("05_ssim_heatmap_overlay.png", overlay)

    # Step 3 — edge density difference (structural change via edge shift)
    edge_block = max(ssim_window // 2, 16)
    edge_mask = _edge_density_difference_map(
        ref_blur, test_blur, edge_low, edge_high,
        edge_block, edge_density_threshold,
    )

    if save_debug:
        _save_debug( "06_edge_mask.png", edge_mask)

    # Step 4 — fuse
    fused = _fuse_maps(ssim_mask, edge_mask, fusion_mode)

    if save_debug:
        _save_debug( "07_fused_mask.png", fused)

    # Step 4b — optional colorfulness pre-mask
    # Suppresses alignment artifacts that affect all channels uniformly.
    colorfulness_mask = None
    if use_colorfulness_mask:
        colorfulness_mask = _compute_colorfulness_mask(
            ref_bgr, test_warped_bgr,
            colorfulness_block_size, colorfulness_ratio_threshold, 15.0,
        )
        fused = fused & colorfulness_mask

        if save_debug:
            _save_debug("08_colorfulness_mask.png", colorfulness_mask)

    # Step 5 — exclude non-common region (black borders, etc.)
    fused = fused & common_mask

    # Step 6 — morphological open: remove small isolated noise
    opened = _morph_open(fused, morph_open_kernel_size)

    # Step 7 — morphological close: fill holes in real diff regions
    closed = _morph_close(opened, morph_close_kernel_size)

    if save_debug:
        _save_debug( "09_cleaned_mask.png", closed)

    # Step 8 — extract bounding boxes
    bboxes = _extract_bboxes(closed, min_area)

    # Step 8b — filter by color-diff significance
    if use_color_ssim and len(bboxes) > 0:
        filtered = _filter_by_color_diff(
            ref_bgr, test_warped_bgr, bboxes,
            min_color_ratio=min_color_ratio,
        )
        if filtered:
            bboxes = filtered

    # Step 8c — filter giant bboxes (web-like false positives from edge networks)
    if max_bbox_image_ratio > 0 and len(bboxes) > 0:
        img_area = float(ref_bgr.shape[0] * ref_bgr.shape[1])
        bboxes = [
            b for b in bboxes
            if (b[2] * b[3]) / img_area <= max_bbox_image_ratio
        ]

    # Step 9 — scale bboxes, mask and ssim_map back to original coordinates
    if ds_scale != 1.0:
        bboxes = [
            (int(b[0] / ds_scale), int(b[1] / ds_scale),
             int(b[2] / ds_scale), int(b[3] / ds_scale))
            for b in bboxes
        ]
        closed = cv2.resize(
            closed.astype(np.uint8), (_orig_w, _orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        ssim_map = cv2.resize(
            ssim_map.astype(np.float32), (_orig_w, _orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return bboxes, closed, ssim_map


def detect_differences_ssim_only(
    ref_bgr: np.ndarray,
    test_warped_bgr: np.ndarray,
    common_mask: np.ndarray,
    *,
    gaussian_blur_sigma: float,
    ssim_window: int,
    min_area: int,
    use_color_ssim: bool,
    max_detection_side: int = 0,
    close_kernel_size: int = 5,
    save_debug: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    """SSIM 热力图 → Otsu 自动阈值 → 闭运算 → 连通域 → 面积过滤。

    管线: 模糊 → SSIM → 相异性图 → Otsu阈值分割
          → 闭运算填洞 → 连通域提取 → min_area筛噪 → 边缘框。
    返回 (bboxes, mask, ssim_map)。
    """
    ref_gray = _gray(ref_bgr)
    test_gray = _gray(test_warped_bgr)

    # 缩放大图
    ds_scale = 1.0
    _orig_h, _orig_w = ref_bgr.shape[:2]
    if max_detection_side > 0:
        max_side = max(ref_bgr.shape[0], ref_bgr.shape[1])
        if max_side > max_detection_side:
            ds_scale = max_detection_side / max_side
            new_w = int(ref_bgr.shape[1] * ds_scale)
            new_h = int(ref_bgr.shape[0] * ds_scale)
            ref_bgr = cv2.resize(ref_bgr, (new_w, new_h))
            test_warped_bgr = cv2.resize(test_warped_bgr, (new_w, new_h))
            common_mask = cv2.resize(
                common_mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            ref_gray = _gray(ref_bgr)
            test_gray = _gray(test_warped_bgr)

    # 模糊 → SSIM
    if use_color_ssim:
        ref_blur_bgr, test_blur_bgr = _preprocess(
            ref_bgr, test_warped_bgr, gaussian_blur_sigma)
        _ref_for_ssim = ref_blur_bgr
        _test_for_ssim = test_blur_bgr
        _ssim_fn = _compute_ssim_map_color
    else:
        ref_blur, test_blur = _preprocess(ref_gray, test_gray, gaussian_blur_sigma)
        _ref_for_ssim = ref_blur
        _test_for_ssim = test_blur
        _ssim_fn = _compute_ssim_map

    # 填非公共区域防边缘伪影
    _test_for_ssim = _test_for_ssim.copy()
    _test_for_ssim[~common_mask] = _ref_for_ssim[~common_mask]
    ssim_map = _ssim_fn(_ref_for_ssim, _test_for_ssim, ssim_window)

    # 相异性图 (0=相同, 255=完全不同)
    dissim = ((1.0 - np.clip(ssim_map, 0, 1)) * 255).astype(np.uint8)

    # Otsu 自动阈值分割
    otsu_thresh, diff_mask = cv2.threshold(
        dissim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if save_debug:
        _save_debug("05a_dissim.png", dissim)
        _save_debug("05b_otsu_thresh.png", diff_mask)

    # 限定公共区域
    diff_mask = (diff_mask > 0) & common_mask

    # 闭运算 — 填洞连接断裂区域
    if close_kernel_size >= 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        diff_mask = cv2.morphologyEx(
            diff_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if save_debug:
        _save_debug("05c_closed.png", diff_mask)

    # 连通域提取 + min_area 过滤
    diff_mask_bool = diff_mask.astype(bool)
    bboxes = _extract_bboxes(diff_mask_bool, min_area)

    # 缩放回原图
    if ds_scale != 1.0:
        bboxes = [
            (int(b[0] / ds_scale), int(b[1] / ds_scale),
             int(b[2] / ds_scale), int(b[3] / ds_scale))
            for b in bboxes
        ]
        diff_mask_bool = cv2.resize(
            diff_mask_bool.astype(np.uint8), (_orig_w, _orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        ssim_map = cv2.resize(
            ssim_map.astype(np.float32), (_orig_w, _orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return bboxes, diff_mask_bool, ssim_map
