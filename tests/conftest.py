import pytest
import numpy as np
import cv2
from pathlib import Path

TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"


@pytest.fixture
def simple_ref():
    """256x256 white image with a black square in corner."""
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    img[50:100, 50:100] = [0, 0, 0]
    return img


@pytest.fixture
def simple_test():
    """Same as ref but missing the black square."""
    return np.full((256, 256, 3), 255, dtype=np.uint8)


@pytest.fixture
def identical_pair():
    """Two identical small images."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    return img, img.copy()


@pytest.fixture
def shifted_pair():
    """Ref and test with a known 5px translation."""
    ref = np.full((256, 256, 3), 128, dtype=np.uint8)
    cv2.rectangle(ref, (80, 80), (120, 120), (255, 255, 255), -1)
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    test = cv2.warpAffine(ref, M, (256, 256))
    return ref, test


@pytest.fixture
def default_params():
    from core.params import DetectionParams
    return DetectionParams()
