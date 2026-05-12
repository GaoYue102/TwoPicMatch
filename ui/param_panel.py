from typing import Optional

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QHBoxLayout, QLabel, QScrollArea, QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal

from core.params import DetectionParams

_FEATURE_TYPES = ["AKAZE", "SIFT", "ORB"]
_FUSION_MODES = ["AND", "OR"]


class ParamPanel(QDockWidget):
    """Dockable parameter-tuning panel."""

    reprocess_requested = pyqtSignal(DetectionParams)
    display_changed = pyqtSignal()              # 仅刷新显示，不重跑管线

    def __init__(self, params: Optional[DetectionParams] = None, parent=None):
        super().__init__("参数面板", parent)
        self._params = params or DetectionParams()
        self._building = True
        self._widgets: dict = {}
        self._setup_ui()
        self._building = False

    # ------------------------------------------------------------------
    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        self.setWidget(scroll)

        layout = QVBoxLayout(container)
        self._ssim_hidable = []  # widgets to hide when SSIM-only is active

        # -- Presets --
        from core.params import PRESETS
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        self._preset_cb = QComboBox()
        self._preset_cb.addItems(list(PRESETS.keys()))
        self._preset_cb.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._preset_cb)
        layout.addLayout(preset_layout)

        # -- Feature Matching --
        grp = QGroupBox("特征匹配")
        form = QFormLayout(grp)

        cb = QComboBox()
        cb.addItems(_FEATURE_TYPES)
        cb.setCurrentText(self._params.feature_type)
        cb.currentTextChanged.connect(self._on_feature_changed)
        form.addRow("算法:", cb)
        self._widgets["feature_type"] = cb

        sl, sp = self._make_float_slider(0.3, 0.99, self._params.feature_threshold, 2)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("Ratio阈值:", self._h_pack(sl, sp))
        self._widgets["feature_threshold"] = (sl, sp)

        sl, sp = self._make_float_slider(0.5, 20.0, self._params.ransac_threshold, 1)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("RANSAC阈值:", self._h_pack(sl, sp))
        self._widgets["ransac_threshold"] = (sl, sp)

        sl, sp = self._make_int_slider(0, 8192, self._params.max_matching_side, step=256)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("匹配缩放上限:", self._h_pack(sl, sp))
        self._widgets["max_matching_side"] = (sl, sp)

        sl, sp = self._make_float_slider(0.0001, 0.005, self._params.akaze_threshold, 4)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("AKAZE阈值:", self._h_pack(sl, sp))
        self._widgets["akaze_threshold"] = (sl, sp)

        sl, sp = self._make_float_slider(0.9, 1.0, self._params.ransac_confidence, 3)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("RANSAC置信度:", self._h_pack(sl, sp))
        self._widgets["ransac_confidence"] = (sl, sp)

        layout.addWidget(grp)

        # -- Illumination --
        grp = QGroupBox("光照归一化")
        form = QFormLayout(grp)

        cb = QCheckBox()
        cb.setChecked(self._params.do_histogram_match)
        cb.toggled.connect(self._on_param_changed)
        form.addRow("直方图匹配:", cb)
        self._widgets["do_histogram_match"] = cb

        sl, sp = self._make_float_slider(0.5, 10.0, self._params.clahe_clip_limit, 1)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("CLAHE阈值:", self._h_pack(sl, sp))
        self._widgets["clahe_clip_limit"] = (sl, sp)

        sl, sp = self._make_int_slider(2, 32, self._params.clahe_tile_size)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("CLAHE分块:", self._h_pack(sl, sp))
        self._widgets["clahe_tile_size"] = (sl, sp)

        layout.addWidget(grp)

        # -- Pre-processing --
        grp = QGroupBox("预处理")
        form = QFormLayout(grp)

        sl, sp = self._make_float_slider(0.0, 5.0, self._params.gaussian_blur_sigma, 1)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("高斯模糊σ:", self._h_pack(sl, sp))
        self._widgets["gaussian_blur_sigma"] = (sl, sp)

        sl, sp = self._make_int_slider(0, 20, self._params.border_margin)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("边界裕量(px):", self._h_pack(sl, sp))
        self._widgets["border_margin"] = (sl, sp)

        layout.addWidget(grp)

        # -- Difference Detection --
        grp = QGroupBox("差异检测")
        form = QFormLayout(grp)

        cb_ssim_only = QCheckBox()
        cb_ssim_only.setChecked(self._params.ssim_only_mode)
        cb_ssim_only.toggled.connect(self._on_param_changed)
        cb_ssim_only.toggled.connect(self._on_ssim_only_toggled)
        form.addRow("SSIM Only:", cb_ssim_only)
        self._widgets["ssim_only_mode"] = cb_ssim_only

        cb_dd = QCheckBox()
        cb_dd.setChecked(self._params.use_direct_diff)
        cb_dd.toggled.connect(self._on_param_changed)
        form.addRow("直接差值融合:", cb_dd)
        self._widgets["use_direct_diff"] = cb_dd

        cb_color = QCheckBox()
        cb_color.setChecked(self._params.use_color_ssim)
        cb_color.toggled.connect(self._on_param_changed)
        form.addRow("颜色感知SSIM:", cb_color)
        self._widgets["use_color_ssim"] = cb_color

        sl, sp = self._make_int_slider(0, 8192, self._params.max_detection_side, step=256)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("检测缩放上限:", self._h_pack(sl, sp))
        self._widgets["max_detection_side"] = (sl, sp)

        sl, sp = self._make_int_slider(16, 256, self._params.ssim_window, step=8)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("SSIM窗口:", self._h_pack(sl, sp))
        self._widgets["ssim_window"] = (sl, sp)

        sl, sp = self._make_float_slider(0.3, 0.99, self._params.ssim_threshold, 2)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("SSIM阈值:", row)
        self._widgets["ssim_threshold"] = (sl, sp)
        self._ssim_hidable.append(row)

        sl, sp = self._make_int_slider(5, 200, self._params.edge_threshold_low)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("Canny低:", row)
        self._widgets["edge_threshold_low"] = (sl, sp)
        self._ssim_hidable.append(row)

        sl, sp = self._make_int_slider(10, 500, self._params.edge_threshold_high)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("Canny高:", row)
        self._widgets["edge_threshold_high"] = (sl, sp)
        self._ssim_hidable.append(row)

        sl, sp = self._make_float_slider(0.01, 0.5, self._params.edge_density_threshold, 3)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("边缘密度:", row)
        self._widgets["edge_density_threshold"] = (sl, sp)
        self._ssim_hidable.append(row)

        sl, sp = self._make_float_slider(1.0, 5.0, self._params.min_color_ratio, 1)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("颜色差异比:", row)
        self._widgets["min_color_ratio"] = (sl, sp)
        self._ssim_hidable.append(row)

        bg = QButtonGroup(self)
        radio_and = QRadioButton("AND (严格)")
        radio_or = QRadioButton("OR (宽松)")
        bg.addButton(radio_and, 0)
        bg.addButton(radio_or, 1)
        if self._params.fusion_mode == "AND":
            radio_and.setChecked(True)
        else:
            radio_or.setChecked(True)
        bg.idToggled.connect(self._on_fusion_changed)
        row = self._h_pack(radio_and, radio_or)
        form.addRow("融合模式:", row)
        self._widgets["fusion_mode"] = bg
        self._ssim_hidable.append(row)

        cb_cfm = QCheckBox()
        cb_cfm.setChecked(self._params.use_colorfulness_mask)
        cb_cfm.toggled.connect(self._on_param_changed)
        form.addRow("色差比预掩膜:", cb_cfm)
        self._widgets["use_colorfulness_mask"] = cb_cfm
        self._ssim_hidable.append(cb_cfm)

        sl, sp = self._make_int_slider(8, 128, self._params.colorfulness_block_size, step=8)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("色差块大小:", row)
        self._widgets["colorfulness_block_size"] = (sl, sp)
        self._ssim_hidable.append(row)

        sl, sp = self._make_float_slider(1.2, 5.0, self._params.colorfulness_ratio_threshold, 1)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("色差比阈值:", row)
        self._widgets["colorfulness_ratio_threshold"] = (sl, sp)
        self._ssim_hidable.append(row)

        layout.addWidget(grp)

        # -- Post-processing --
        grp = QGroupBox("后处理")
        form = QFormLayout(grp)

        sl, sp = self._make_int_slider(0, 15, self._params.morph_open_kernel_size, step=2)
        sl.valueChanged.connect(self._on_param_changed)
        row = self._h_pack(sl, sp)
        form.addRow("开运算核:", row)
        self._widgets["morph_open_kernel_size"] = (sl, sp)
        self._ssim_hidable.append(row)

        sl, sp = self._make_int_slider(1, 31, self._params.morph_close_kernel_size, step=2)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("闭运算核:", self._h_pack(sl, sp))
        self._widgets["morph_close_kernel_size"] = (sl, sp)

        sl, sp = self._make_int_slider(10, 5000, self._params.min_area, step=10)
        sl.valueChanged.connect(self._on_param_changed)
        form.addRow("最小面积:", self._h_pack(sl, sp))
        self._widgets["min_area"] = (sl, sp)

        layout.addWidget(grp)

        # -- Visualization --
        grp = QGroupBox("可视化")
        form = QFormLayout(grp)

        cb_hm = QCheckBox()
        cb_hm.setChecked(self._params.show_heatmap)
        cb_hm.toggled.connect(self._on_param_changed)
        cb_hm.toggled.connect(lambda: self.display_changed.emit())
        form.addRow("热力图叠加:", cb_hm)
        self._widgets["show_heatmap"] = cb_hm

        sl, sp = self._make_float_slider(0.1, 1.0, self._params.heatmap_opacity, 2)
        sl.valueChanged.connect(self._on_param_changed)
        sl.valueChanged.connect(lambda v: self.display_changed.emit())
        form.addRow("热力图透明度:", self._h_pack(sl, sp))
        self._widgets["heatmap_opacity"] = (sl, sp)

        layout.addWidget(grp)

        # -- Debug --
        cb_debug = QCheckBox("导出处理步骤图 (debug_steps/)")
        cb_debug.setChecked(self._params.debug_export)
        cb_debug.toggled.connect(self._on_param_changed)
        layout.addWidget(cb_debug)
        self._widgets["debug_export"] = cb_debug

        # -- Buttons --
        btn_layout = QHBoxLayout()
        btn_run = QPushButton("重新检测")
        btn_run.clicked.connect(self._on_reprocess)
        btn_layout.addWidget(btn_run)
        layout.addLayout(btn_layout)
        layout.addStretch()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _h_pack(self, *widgets):
        w = QWidget()
        lo = QHBoxLayout(w)
        lo.setContentsMargins(0, 0, 0, 0)
        for child in widgets:
            lo.addWidget(child)
        return w

    def _make_float_slider(self, lo, hi, init, decimals):
        slider = QSlider(Qt.Orientation.Horizontal)
        mul = 10 ** decimals
        slider.setRange(int(lo * mul), int(hi * mul))
        slider.setValue(int(init * mul))

        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** (1 - decimals) if decimals > 1 else 0.1)
        spin.setValue(init)
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * mul)))
        slider.valueChanged.connect(lambda v: spin.setValue(v / mul))
        return slider, spin

    def _make_int_slider(self, lo, hi, init, step=1):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(init)
        slider.setSingleStep(step)

        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(init)
        spin.setSingleStep(step)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(spin.setValue)
        return slider, spin

    # ------------------------------------------------------------------
    # slots
    # ------------------------------------------------------------------
    def _on_feature_changed(self, txt):
        self._params.feature_type = txt

    def _on_preset_changed(self, name):
        if self._building:
            return
        from core.params import params_with_preset, DetectionParams
        self._params = params_with_preset(DetectionParams(), name)
        self.set_params(self._params)
        self.reprocess_requested.emit(self._params)

    def _on_fusion_changed(self, idx, _checked):
        if not _checked:
            return
        self._params.fusion_mode = _FUSION_MODES[idx]

    def _on_param_changed(self):
        if self._building:
            return
        # int sliders / spinners
        int_keys = [
            "ssim_window", "edge_threshold_low", "edge_threshold_high",
            "clahe_tile_size", "border_margin",
            "morph_open_kernel_size", "morph_close_kernel_size", "min_area",
            "max_detection_side", "max_matching_side",
            "colorfulness_block_size",
        ]
        for key in int_keys:
            w = self._widgets.get(key)
            if w:
                setattr(self._params, key, w[1].value())
        # float sliders
        float_keys = [
            "feature_threshold", "ransac_threshold", "clahe_clip_limit",
            "gaussian_blur_sigma", "ssim_threshold", "edge_density_threshold",
            "min_color_ratio", "colorfulness_ratio_threshold",
            "akaze_threshold", "ransac_confidence", "heatmap_opacity",
        ]
        for key in float_keys:
            w = self._widgets.get(key)
            if w:
                setattr(self._params, key, w[1].value())
        # checkboxes
        for cb_key in ["do_histogram_match", "use_color_ssim", "use_colorfulness_mask",
                       "show_heatmap", "ssim_only_mode", "use_direct_diff",
                       "debug_export"]:
            cb = self._widgets.get(cb_key)
            if cb:
                setattr(self._params, cb_key, cb.isChecked())

    def _on_ssim_only_toggled(self, checked):
        for w in self._ssim_hidable:
            w.setVisible(not checked)

    def _on_reprocess(self):
        self.reprocess_requested.emit(self._params)

    # ------------------------------------------------------------------
    def get_params(self) -> DetectionParams:
        return self._params

    def set_params(self, p: DetectionParams):
        self._params = p
        self._building = True
        for key in p.__dataclass_fields__:
            w = self._widgets.get(key)
            if w is None:
                continue
            val = getattr(p, key)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(val))
            elif isinstance(w, QButtonGroup):
                idx = 0 if str(val) == "AND" else 1
                w.button(idx).setChecked(True)
            elif isinstance(w, tuple) and len(w) == 2:
                w[1].setValue(val)
        self._building = False
        self._on_ssim_only_toggled(p.ssim_only_mode)
