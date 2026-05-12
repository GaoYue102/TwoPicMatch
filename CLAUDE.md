# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

TwoPicMatch 是一个工业装配差异检测工具，用于对比两张图片（参考图 vs 检测图），自动找出装配体中的缺失/错误/移位零件。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 GUI
python main.py

# 运行全部测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_difference_detector.py -v

# 运行单个测试用例
pytest tests/test_difference_detector.py::test_identical_images_no_detections -v

# 端到端回归验证（三个测试图片集）
python -c "
import sys; sys.path.insert(0,'.')
from core.params import DetectionParams
from core.image_io import imread_unicode
from core.feature_matching import compute_feature_match
from core.alignment import compute_alignment
from core.normalization import normalize_illumination
from core.difference_detector import detect_differences

for name, rp, tp in [
    ('simple','test_images/ref_simple.png','test_images/test_simple_diff.png'),
    ('assembly','test_images/ref_assembly.png','test_images/test_missing_parts.png'),
    ('earplug','微信图片_20260121092556_28_61.jpg','微信图片_20260121092600_29_61.jpg'),
]:
    params = DetectionParams()
    ref = imread_unicode(rp); test = imread_unicode(tp)
    fm = compute_feature_match(ref, test, params.feature_type, params.feature_threshold, params.ransac_threshold)
    assert fm.H is not None, f'{name}: HOMOG FAIL'
    warped, mask = compute_alignment(ref, test, fm.H, params.border_margin)
    ref_norm, test_norm = normalize_illumination(warped, ref, mask, do_hist_match=params.do_histogram_match, clahe_clip=params.clahe_clip_limit, clahe_tile=params.clahe_tile_size)
    bboxes, dm = detect_differences(ref_norm, test_norm, mask, gaussian_blur_sigma=params.gaussian_blur_sigma, ssim_window=params.ssim_window, ssim_threshold=params.ssim_threshold, edge_low=params.edge_threshold_low, edge_high=params.edge_threshold_high, edge_density_threshold=params.edge_density_threshold, fusion_mode=params.fusion_mode, morph_open_kernel_size=params.morph_open_kernel_size, morph_close_kernel_size=params.morph_close_kernel_size, min_area=params.min_area, use_color_ssim=params.use_color_ssim, max_detection_side=params.max_detection_side, min_color_ratio=params.min_color_ratio)
    assert len(bboxes) > 0, f'{name}: ZERO DETECTIONS'
    print(f'{name}: {len(bboxes)} bboxes OK')
print('ALL PASSED')
"
```

## 核心架构 — 检测管线

管线在 `workers/pipeline_worker.py:66` 的 `_run_pipeline()` 中依次执行 6 个阶段：

```
加载图像 → 特征匹配 → 透视对齐 → 光照归一化 → 差异检测 → 叠加结果
```

### Stage 1-2: 匹配与对齐
- `core/feature_matching.py` — 核心入口是 `compute_feature_match()`，一次调用完成 detect + match + homography，返回 `FeatureMatchResult`（包含 kp1/kp2/des1/des2/good_matches/H/stats）。`find_homography()` 是其向后兼容的 thin wrapper。
- 大图先缩放到 `max_matching_side`(默认2048) 做匹配，得到的 H 矩阵再换算回原图尺寸
- `core/alignment.py` — `compute_alignment()` 用 H 对 test 图做透视变换，同时生成公共区域 mask（排除黑边和插值伪影）
- `validate_homography()` 检查内点比、缩放/剪切极端值、行列式（镜面/退化）

### Stage 3: 光照归一化
- `core/normalization.py` — `normalize_illumination()` 对 **ref 和 test 双方**做 CLAHE（LAB 空间 L 通道），可选直方图匹配（默认关）
- 返回 `(ref_norm, test_norm)`，不修改原图

### Stage 4: 差异检测（核心算法）
`core/difference_detector.py:163` `detect_differences()`：

1. **缩放** — 若 `max_detection_side > 0` 且图超过该尺寸，等比缩放到该边界
2. **色彩感知 SSIM** — 若 `use_color_ssim=True`，对 B/G/R 三通道分别算 SSIM 取 min，能捕获灰度 SSIM 不可见的颜色变化（如橙色耳塞）
3. **边缘密度差异** — Canny + boxFilter 局域密度对比
4. **融合** — "AND"（两者都满足）或 "OR"（任一满足）
5. **形态学** — 开运算去噪 → 闭运算填洞
6. **连通域提取** — `connectedComponentsWithStats`，过滤 `min_area`
7. **颜色差异后过滤** — 对每个候选框计算 B/G/R 通道差异比，比例 ≥ `min_color_ratio`（默认 1.5）才保留；全被滤掉则回退保留原始框

**Step 4b — 色差比预掩膜（可选）**
- 对齐噪声均匀影响所有 BGR 通道（ratio≈1.0），真实物体变化导致通道失衡（ratio>>1.0）
- 块级分析：将图像分块，每块计算通道差异比，超过 `colorfulness_ratio_threshold`(默认 2.0) 的块保留
- 适用于手持拍照/视差严重的图像对
- 开启后可能减少检出数（过滤伪影），但同时能发现被噪声淹没的真实差异

### Stage 5: 匹配可视化
- 复用 `FeatureMatchResult` 通过 `draw_matches_from_result()` 生成，不再重复计算特征

## 参数系统

所有可调参数集中在 `core/params.py` 的 `DetectionParams` dataclass（27 个字段），带 `to_dict()`/`from_dict()` 序列化。四个预设（默认/严格/宽松/高速）在 `PRESETS` 字典中。

**关键参数速查：**

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `feature_type` | AKAZE | 特征算法 (AKAZE/SIFT/ORB) |
| `feature_threshold` | 0.7 | Lowe's ratio 阈值 (↓更严格) |
| `ransac_threshold` | 5.0 | RANSAC 重投影误差阈值 (px) |
| `ssim_window` | 48 | SSIM 窗口大小 (↑更容忍对不齐) |
| `ssim_threshold` | 0.88 | SSIM 低于此值即标记差异 |
| `use_color_ssim` | True | 启用逐通道颜色 SSIM |
| `fusion_mode` | AND | SSIM与边缘的融合方式 |
| `min_color_ratio` | 1.5 | 颜色过滤比阈值 (↑更严格) |
| `min_area` | 500 | 最小差异区域面积 (px²) |
| `max_bbox_image_ratio` | 0.5 | 过滤掉超过图像 50% 的大框 |
| `gaussian_blur_sigma` | 1.5 | 预模糊程度 (↓更敏感) |
| `use_colorfulness_mask` | False | 块级色差比预掩膜（手持拍照场景开启） |
| `colorfulness_ratio_threshold` | 2.0 | 通道差异比阈值 (↓检出更多) |
| `colorfulness_block_size` | 32 | 色差分析块大小 (px) |

## UI 架构

- `main.py` → `ui/main_window.py` `MainWindow`
- 中央：`ui/dual_viewer.py` 左右分屏双画布（`ui/image_canvas.py`），缩放/平移同步
- 右侧 dock：`ui/param_panel.py`（参数调节）+ `ui/result_list.py`（差异结果列表，点击跳转）
- 叠加层：`ui/result_overlay.py` 在双画布上绘制红色差异框
- 后台线程：`workers/pipeline_worker.py` `PipelineWorker(QThread)`，通过 signal 与 UI 通信，支持 `requestInterruption()` 取消
- 参数持久化：`QSettings("TwoPicMatch", "TwoPicMatch")` 在 `closeEvent` 自动保存，启动时恢复

## 测试

`tests/` 目录包含 18 个 pytest 用例：
- `conftest.py` — 合成图像 fixtures（identical_pair, shifted_pair, simple_ref/test）
- `test_params.py` — 参数序列化、预设应用
- `test_feature_matching.py` — 特征匹配、单应性验证、match viz
- `test_difference_detector.py` — 差异检测正确性（相同→0检出、明显差异→≥1检出、颜色vs灰度、缩放一致性）
- `test_integration.py` — 使用 `test_images/` 真实图片的端到端测试

## 注意事项

- 所有图像路径必须用 `core/image_io.py` 的 `imread_unicode()` 读取，原生 `cv2.imread()` 不支持中文路径
- 特征检测和差异检测各有一个独立的缩放上限参数（`max_matching_side` / `max_detection_side`），大图场景下两者可独立调节
- `normalize_illumination` 返回的 ref 是 CLAHE 处理后的副本，原始 `r.ref_bgr` 保持不变，导出功能使用原图
- `test_images/` 中的简单图案是网格状色块，是边缘检测的极端测试用例——网格线轻微不对齐就会产生大量伪影
- `ui/param_panel.py` 的 `_on_param_changed` 使用了硬编码的参数名列表（`int_keys`/`float_keys`/checkbox_keys），新增参数需同步添加到对应列表
- 手持拍照/视差严重的场景（如 earplug 测试集），应开启「色差比预掩膜」(`use_colorfulness_mask=True`)；固定相机拍摄的场景保持关闭以避免过滤掉微弱但真实的差异

## 技术栈

- Python 3.9+，依赖：PyQt6、opencv-contrib-python、numpy、scikit-image
- 图像路径必须用 `core/image_io.py` 的 `imread_unicode()` 读取（底层 `np.fromfile` + `cv2.imdecode` 模式），原生 `cv2.imread()` 不支持中文路径
- PyQt6 枚举在 `QtCore.Qt` / `QtWidgets` 命名空间下（不是 Qt5 路径）

## Bug 修复与验证

修改检测参数后，必须跑端到端回归脚本验证三个测试图片集（simple, assembly, earplug），对比基线检出数。检出数下降超过 20% 则自动回退参数修改。可用 `/cv-debug` 技能一键执行回归验证和诊断。

修改任何代码后：
- 运行 `pytest tests/ -v` 确保全部 18 个测试通过
- 如有 Bug 修复，跑回归脚本展示修前/修后的检出数对比
- 不要声称修好但没有验证输出

## 响应风格

当被问及实现指导（如何构建/复现某功能）时，直接给出编号步骤 + 代码块 + 命令，不要先讲架构原理。以 "Step 1: ..." 格式开始。只在用户明确要求时才解释架构。