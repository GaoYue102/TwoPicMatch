from typing import Optional

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QPainter


class ImageCanvas(QGraphicsView):
    """Zoomable / pannable image view. Emits zoom_updated for cross-view syncing."""

    zoom_updated = pyqtSignal(float, QPointF)  # scale_factor, scene_center
    pan_updated = pyqtSignal(QPointF)  # scene_center

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._source_pixmap: Optional[QPixmap] = None

        self._zoom_factor = 1.15
        self._min_scale = 0.01
        self._max_scale = 20.0
        self._current_scale = 1.0

        self._panning = False
        self._pan_start = QPointF()

        # rendering hints
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    @property
    def scene(self) -> QGraphicsScene:
        return self._scene

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def set_image(self, pixmap: QPixmap):
        self._source_pixmap = pixmap
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._current_scale = 1.0
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        QTimer.singleShot(0, self._emit_zoom_sync)

    def clear_image(self):
        self._scene.clear()
        self._pixmap_item = None
        self._source_pixmap = None
        self._current_scale = 1.0

    def scene_center(self) -> QPointF:
        """Visible rect centre in scene coordinates."""
        return self.mapToScene(self.viewport().rect().center())

    def sync_pan(self, center_scene: QPointF):
        """Apply pan from external source."""
        self.centerOn(center_scene)

    def center_on_bbox(self, x: int, y: int, w: int, h: int):
        """Center the view on a bounding box with some padding."""
        pad = max(w, h) * 0.3
        rect = QRectF(x - pad, y - pad, w + 2 * pad, h + 2 * pad)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_scale = self.transform().m11()
        QTimer.singleShot(0, self._emit_zoom_sync)

    def _apply_zoom_with_absolute(self, target_scale: float, center_scene: QPointF, *, emit: bool):
        """Match an absolute scale factor (used by dual_viewer sync)."""
        if self._current_scale == 0:
            return
        factor = target_scale / self._current_scale
        self._apply_zoom(factor, center_scene, emit=emit)

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self._zoom_factor if delta > 0 else 1.0 / self._zoom_factor
        center_scene = self.mapToScene(event.position().toPoint())
        self._apply_zoom(factor, center_scene, emit=True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            # move by delta in view coords (invert for natural drag)
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            self.pan_updated.emit(self.scene_center())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._emit_zoom_sync()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _apply_zoom(self, factor: float, center_scene: QPointF, emit: bool):
        new_scale = self._current_scale * factor
        if new_scale < self._min_scale or new_scale > self._max_scale:
            return
        self._current_scale = new_scale
        self.scale(factor, factor)
        # re-center on the same scene point
        new_center_view = self.mapFromScene(center_scene)
        delta = self.viewport().rect().center() - new_center_view
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() - delta.x()
        )
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        if emit:
            self.zoom_updated.emit(self._current_scale, self.scene_center())

    def _emit_zoom_sync(self):
        """Emit current state so a paired canvas can match it."""
        self.zoom_updated.emit(self._current_scale, self.scene_center())

