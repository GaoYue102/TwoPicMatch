import numpy as np
import cv2


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread replacement that handles non-ASCII paths on Windows."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise OSError(f"无法读取图像: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    """cv2.imwrite replacement that handles non-ASCII paths on Windows."""
    ext = path.rsplit(".", 1)[-1] if "." in path else ".png"
    success, data = cv2.imencode("." + ext, img, params if params else [])
    if not success:
        return False
    with open(path, "wb") as f:
        f.write(data.tobytes())
    return True


def validate_image_pair(ref: np.ndarray, test: np.ndarray) -> list:
    """Validate that two images are a reasonable pair for comparison.

    Returns list of warning messages (empty list = all good).
    """
    warnings = []

    if ref.ndim != test.ndim:
        warnings.append(
            f"通道数不一致: ref={ref.ndim}维, test={test.ndim}维"
        )
    elif ref.ndim == 3 and ref.shape[2] != test.shape[2]:
        warnings.append(
            f"通道数不一致: ref={ref.shape[2]}, test={test.shape[2]}"
        )

    h_ratio = ref.shape[0] / max(test.shape[0], 1)
    w_ratio = ref.shape[1] / max(test.shape[1], 1)
    if max(h_ratio, 1 / h_ratio) > 4.0 or max(w_ratio, 1 / w_ratio) > 4.0:
        warnings.append(
            f"尺寸差异过大: ref={ref.shape[1]}x{ref.shape[0]}, "
            f"test={test.shape[1]}x{test.shape[0]}"
        )

    min_dim = min(ref.shape[0], ref.shape[1], test.shape[0], test.shape[1])
    if min_dim < 50:
        warnings.append(f"图像过小 (最小维度={min_dim}px)")

    if ref.std() < 1.0:
        warnings.append("参考图疑似空白/纯色")
    if test.std() < 1.0:
        warnings.append("检测图疑似空白/纯色")

    return warnings


def images_are_identical(ref: np.ndarray, test: np.ndarray) -> bool:
    """Quick check if two images are bitwise identical."""
    if ref.shape != test.shape:
        return False
    return bool(np.all(ref == test))
