from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPixmap

from ui.image_canvas import ImageCanvas


class DualViewer(QWidget):
    """Side-by-side image viewers with synchronized zoom & pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ref_canvas = ImageCanvas()
        self._test_canvas = ImageCanvas()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._ref_canvas)
        splitter.addWidget(self._test_canvas)
        splitter.setSizes([500, 500])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # cross-wire zoom & pan
        self._syncing = False

        self._ref_canvas.zoom_updated.connect(self._on_ref_zoom)
        self._ref_canvas.pan_updated.connect(self._on_ref_pan)
        self._test_canvas.zoom_updated.connect(self._on_test_zoom)
        self._test_canvas.pan_updated.connect(self._on_test_pan)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def set_reference(self, pixmap: QPixmap):
        self._ref_canvas.set_image(pixmap)

    def set_test(self, pixmap: QPixmap):
        self._test_canvas.set_image(pixmap)

    def clear_all(self):
        self._ref_canvas.clear_image()
        self._test_canvas.clear_image()

    @property
    def ref_canvas(self) -> ImageCanvas:
        return self._ref_canvas

    @property
    def test_canvas(self) -> ImageCanvas:
        return self._test_canvas

    # ------------------------------------------------------------------
    # sync handlers
    # ------------------------------------------------------------------
    def _on_ref_zoom(self, scale_factor: float, center: QPointF):
        if self._syncing:
            return
        self._syncing = True
        # scale factor is absolute — pass as-is; sync_zoom computes relative delta
        self._test_canvas._apply_zoom_with_absolute(scale_factor, center, emit=False)
        self._syncing = False

    def _on_test_zoom(self, scale_factor: float, center: QPointF):
        if self._syncing:
            return
        self._syncing = True
        self._ref_canvas._apply_zoom_with_absolute(scale_factor, center, emit=False)
        self._syncing = False

    def _on_ref_pan(self, center: QPointF):
        if self._syncing:
            return
        self._syncing = True
        self._test_canvas.centerOn(center)
        self._syncing = False

    def _on_test_pan(self, center: QPointF):
        if self._syncing:
            return
        self._syncing = True
        self._ref_canvas.centerOn(center)
        self._syncing = False
