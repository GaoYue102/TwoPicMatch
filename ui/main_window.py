import os
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSize, QSettings
from PyQt6.QtGui import QAction, QIcon, QPixmap, QImage
from PyQt6.QtWidgets import (
    QMainWindow, QToolBar, QStatusBar, QFileDialog, QMessageBox,
    QApplication,
)

from core.image_io import imread_unicode, imwrite_unicode
from core.params import DetectionParams
from ui.dual_viewer import DualViewer
from ui.param_panel import ParamPanel
from ui.result_overlay import ResultOverlay
from ui.result_list import ResultList
from workers.pipeline_worker import PipelineWorker, PipelineResult


def _cv2_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    if cv_img.ndim == 2:
        h, w = cv_img.shape
        qimg = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, ch = cv_img.shape
        qimg = QImage(cv_img.data, w, h, ch * w, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TwoPicMatch — 双图差异检测")
        self.resize(1400, 900)

        self._params = DetectionParams()
        self._ref_path: Optional[str] = None
        self._test_path: Optional[str] = None
        self._ref_bgr: Optional[np.ndarray] = None
        self._test_bgr: Optional[np.ndarray] = None
        self._last_result: Optional[PipelineResult] = None
        self._worker: Optional[PipelineWorker] = None

        self._dual_viewer = DualViewer()
        self._overlay = ResultOverlay()
        self._param_panel = ParamPanel(self._params)
        self._result_list = ResultList()

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_layout()
        self._connect_signals()
        self._load_settings()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _setup_menubar(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("文件(&F)")
        file_menu.addAction("加载参考图(&R)…", self._load_ref)
        file_menu.addAction("加载检测图(&T)…", self._load_test)
        file_menu.addSeparator()
        file_menu.addAction("导出结果(&E)…", self._export_results)
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close)

        view_menu = mb.addMenu("视图(&V)")
        self._act_show_diffs = QAction("显示差异框", self)
        self._act_show_diffs.setCheckable(True)
        self._act_show_diffs.setChecked(True)
        self._act_show_diffs.toggled.connect(self._overlay.set_visible)
        view_menu.addAction(self._act_show_diffs)

    def _setup_toolbar(self):
        tb = QToolBar("工具栏")
        tb.setIconSize(QSize(24, 24))
        tb.addAction("加载参考图", self._load_ref)
        tb.addAction("加载检测图", self._load_test)
        tb.addSeparator()
        self._act_detect = tb.addAction("开始检测", self._run_detection)
        self._act_detect.setEnabled(False)
        self._act_cancel = tb.addAction("取消检测", self._cancel_detection)
        self._act_cancel.setEnabled(False)
        tb.addSeparator()
        tb.addAction("导出结果", self._export_results)
        self.addToolBar(tb)

    def _setup_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = self._status.showMessage("就绪", 0)
        self._status.showMessage("就绪")

    def _setup_layout(self):
        self.setCentralWidget(self._dual_viewer)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._param_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._result_list)
        self.tabifyDockWidget(self._param_panel, self._result_list)
        self._param_panel.raise_()

    def _connect_signals(self):
        self._param_panel.reprocess_requested.connect(self._on_reprocess)
        self._param_panel.display_changed.connect(self._refresh_display)
        self._result_list.bbox_selected.connect(self._on_bbox_selected)

    def _load_settings(self):
        settings = QSettings("TwoPicMatch", "TwoPicMatch")
        saved = settings.value("params")
        if saved is not None:
            try:
                self._params = DetectionParams.from_dict(saved)
                self._param_panel.set_params(self._params)
            except Exception:
                pass

    def _save_settings(self):
        settings = QSettings("TwoPicMatch", "TwoPicMatch")
        settings.setValue("params", self._params.to_dict())

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # file actions
    # ------------------------------------------------------------------
    def _load_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载参考图", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
        )
        if not path:
            return
        try:
            self._ref_bgr = imread_unicode(path)
            self._ref_path = path
        except OSError as e:
            QMessageBox.warning(self, "错误", str(e))
            self._ref_path = None
            return
        self._dual_viewer.set_reference(_cv2_to_qpixmap(self._ref_bgr))
        self._update_detect_state()
        self._status.showMessage(
            f"参考图: {os.path.basename(path)}  "
            f"({self._ref_bgr.shape[1]}×{self._ref_bgr.shape[0]})"
        )

    def _load_test(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载检测图", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
        )
        if not path:
            return
        try:
            self._test_bgr = imread_unicode(path)
            self._test_path = path
        except OSError as e:
            QMessageBox.warning(self, "错误", str(e))
            self._test_path = None
            return
        self._dual_viewer.set_test(_cv2_to_qpixmap(self._test_bgr))
        self._update_detect_state()
        self._status.showMessage(
            f"检测图: {os.path.basename(path)}  "
            f"({self._test_bgr.shape[1]}×{self._test_bgr.shape[0]})"
        )

    def _update_detect_state(self):
        self._act_detect.setEnabled(
            self._ref_bgr is not None and self._test_bgr is not None
        )

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def _run_detection(self):
        if self._ref_bgr is None or self._test_bgr is None:
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "检测中", "检测正在进行中，请等待完成。")
            return
        self._act_detect.setEnabled(False)
        self._act_cancel.setEnabled(True)
        self._status.showMessage("检测中…")

        self._worker = PipelineWorker(
            self._ref_path, self._test_path, self._param_panel.get_params()
        )
        self._worker.stage_changed.connect(self._on_stage)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel_detection(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._status.showMessage("正在取消…")

    def _on_stage(self, msg: str):
        self._status.showMessage(msg)

    def _on_progress(self, val: float):
        self._status.showMessage(f"检测中… {int(val * 100)}%")

    def _on_finished(self, result: PipelineResult):
        self._last_result = result
        self._act_detect.setEnabled(True)
        self._act_cancel.setEnabled(False)
        self._worker = None

        if result.bboxes:
            self._result_list.set_bboxes(result.bboxes)
        else:
            self._result_list.clear()

        self._refresh_display()

        # release arrays not needed for display refresh
        result.diff_mask = None
        result.match_viz = None

        stats = result.stats
        msg = (
            f"完成 — 差异: {stats.get('num_diffs', 0)} 处  |  "
            f"匹配: {stats.get('num_matches', 0)} 对  "
            f"内点: {stats.get('num_inliers', 0)}"
            f"({stats.get('inlier_ratio', 0):.1%})"
        )
        self._status.showMessage(msg)

    def _on_failed(self, err: str):
        self._act_detect.setEnabled(True)
        self._act_cancel.setEnabled(False)
        self._overlay.clear()
        self._result_list.clear()
        self._last_result = None
        self._worker = None
        self._status.showMessage(f"失败: {err}")
        QMessageBox.critical(self, "检测失败", err)

    def _on_bbox_selected(self, x: int, y: int, w: int, h: int):
        self._dual_viewer.ref_canvas.center_on_bbox(x, y, w, h)
        self._dual_viewer.test_canvas.center_on_bbox(x, y, w, h)

    def _on_reprocess(self, params: DetectionParams):
        self._params = params
        if self._ref_bgr is not None:
            self._dual_viewer.set_reference(_cv2_to_qpixmap(self._ref_bgr))
        if self._test_bgr is not None:
            self._dual_viewer.set_test(_cv2_to_qpixmap(self._test_bgr))
        self._run_detection()

    # ------------------------------------------------------------------
    # heatmap overlay
    # ------------------------------------------------------------------
    def _make_heatmap_overlay(self, base_img: np.ndarray,
                              ssim_map: np.ndarray,
                              common_mask: np.ndarray,
                              opacity: float) -> np.ndarray:
        h, w = base_img.shape[:2]
        if ssim_map.shape[:2] != (h, w):
            sm = cv2.resize(ssim_map.astype(np.float32), (w, h),
                            interpolation=cv2.INTER_LINEAR)
        else:
            sm = ssim_map
        if common_mask.shape[:2] != (h, w):
            cm = cv2.resize(common_mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            cm = common_mask
        ssim_dissim = ((1.0 - np.clip(sm, 0, 1)) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(ssim_dissim, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(heatmap, opacity, base_img, 1.0 - opacity, 0)
        result = base_img.copy()
        result[cm] = blended[cm]
        return result

    def _refresh_display(self):
        r = self._last_result
        if r is None:
            return

        show = self._params.show_heatmap
        opacity = self._params.heatmap_opacity
        has_heatmap = (show and r.ssim_map is not None
                       and r.common_mask is not None)

        # left — ref image
        if has_heatmap and r.ref_norm is not None:
            ref_disp = self._make_heatmap_overlay(
                r.ref_norm, r.ssim_map, r.common_mask, opacity)
        else:
            ref_disp = r.ref_bgr if r.ref_bgr is not None else r.ref_norm
        self._dual_viewer.set_reference(_cv2_to_qpixmap(ref_disp))

        # right — test (warped) image
        if r.test_warped is not None:
            if has_heatmap:
                test_disp = self._make_heatmap_overlay(
                    r.test_warped, r.ssim_map, r.common_mask, opacity)
            else:
                test_disp = r.test_warped
            self._dual_viewer.set_test(_cv2_to_qpixmap(test_disp))

        # re-apply overlays — 优先自由轮廓
        ref_scene = self._dual_viewer.ref_canvas.scene
        test_scene = self._dual_viewer.test_canvas.scene
        if r.contours:
            self._overlay.set_contours(r.contours, ref_scene, test_scene)
        elif r.bboxes:
            self._overlay.set_bboxes(r.bboxes, ref_scene, test_scene)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def _export_results(self):
        if self._last_result is None or self._last_result.ref_bgr is None:
            QMessageBox.information(self, "导出", "请先完成一次检测。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "result.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return

        r = self._last_result
        out = r.ref_bgr.copy()
        if r.contours:
            cv2.drawContours(out, r.contours, -1, (0, 0, 255), 3)
        else:
            for (x, y, w, h) in r.bboxes:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)

        imwrite_unicode(path, out)
        self._status.showMessage(f"结果已导出: {os.path.basename(path)}")
