from typing import List, Tuple

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsScene


_DIFF_COLOR = QColor(255, 60, 60, 200)
_DIFF_PEN = QPen(_DIFF_COLOR, 3)


class ResultOverlay:
    """Manages difference bounding-box overlay items on graphics scenes."""

    def __init__(self):
        self._ref_items: List[QGraphicsRectItem] = []
        self._test_items: List[QGraphicsRectItem] = []

    def set_bboxes(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        ref_scene: QGraphicsScene,
        test_scene: QGraphicsScene,
    ):
        self.clear()
        for (x, y, w, h) in bboxes:
            rect = QRectF(float(x), float(y), float(w), float(h))

            ref_item = QGraphicsRectItem(rect)
            ref_item.setPen(_DIFF_PEN)
            ref_scene.addItem(ref_item)
            self._ref_items.append(ref_item)

            test_item = QGraphicsRectItem(rect)
            test_item.setPen(_DIFF_PEN)
            test_scene.addItem(test_item)
            self._test_items.append(test_item)

    def clear(self):
        for item in self._ref_items + self._test_items:
            try:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            except RuntimeError:
                pass  # C++ object already destroyed by scene.clear()
        self._ref_items.clear()
        self._test_items.clear()

    def set_visible(self, visible: bool):
        for item in self._ref_items + self._test_items:
            try:
                item.setVisible(visible)
            except RuntimeError:
                pass
