from typing import List, Tuple

import numpy as np
from PyQt6.QtCore import QRectF, QPointF
from PyQt6.QtGui import QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene


_DIFF_COLOR = QColor(255, 60, 60, 200)


class ResultOverlay:
    """Manages difference overlay items (rects or free-form contours) on scenes."""

    def __init__(self):
        self._ref_items: list = []
        self._test_items: list = []

    def set_bboxes(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        ref_scene: QGraphicsScene,
        test_scene: QGraphicsScene,
        img_width: int = 1000,
    ):
        self.clear()
        pen = QPen(_DIFF_COLOR, max(1, int(img_width * 0.004)))
        for (x, y, w, h) in bboxes:
            rect = QRectF(float(x), float(y), float(w), float(h))

            ref_item = QGraphicsRectItem(rect)
            ref_item.setPen(pen)
            ref_scene.addItem(ref_item)
            self._ref_items.append(ref_item)

            test_item = QGraphicsRectItem(rect)
            test_item.setPen(pen)
            test_scene.addItem(test_item)
            self._test_items.append(test_item)

    def set_contours(
        self,
        contours: List[np.ndarray],
        ref_scene: QGraphicsScene,
        test_scene: QGraphicsScene,
        img_width: int = 1000,
    ):
        """Draw free-form contours as polygons.

        img_width is used to auto-size the pen: max(1, int(img_width * 0.004)).
        """
        self.clear()
        pen_width = max(1, int(img_width * 0.004))
        pen = QPen(_DIFF_COLOR, pen_width)
        for cnt in contours:
            pts = cnt[:, 0, :]  # (N, 1, 2) → (N, 2)
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in pts])

            ref_item = QGraphicsPolygonItem(poly)
            ref_item.setPen(pen)
            ref_scene.addItem(ref_item)
            self._ref_items.append(ref_item)

            test_item = QGraphicsPolygonItem(poly)
            test_item.setPen(pen)
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
