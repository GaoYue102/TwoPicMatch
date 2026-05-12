class TestDetectionParams:
    def test_default_to_dict_from_dict_roundtrip(self):
        from core.params import DetectionParams
        p1 = DetectionParams()
        d = p1.to_dict()
        p2 = DetectionParams.from_dict(d)
        assert p1 == p2

    def test_from_dict_ignores_unknown_keys(self):
        from core.params import DetectionParams
        p = DetectionParams.from_dict({"ssim_window": 96, "unknown_key": 123})
        assert p.ssim_window == 96

    def test_presets_exist(self):
        from core.params import PRESETS
        assert "默认" in PRESETS
        assert "严格 (高精度)" in PRESETS
        assert "宽松 (高召回)" in PRESETS
        assert "高速模式" in PRESETS

    def test_preset_application(self):
        from core.params import DetectionParams, params_with_preset
        base = DetectionParams()
        strict = params_with_preset(base, "严格 (高精度)")
        assert strict.ssim_threshold == 0.92
        assert strict.fusion_mode == "AND"
        assert base.ssim_threshold != strict.ssim_threshold  # base unchanged

    def test_all_fields_in_to_dict(self):
        from core.params import DetectionParams
        p = DetectionParams()
        d = p.to_dict()
        for field_name in p.__dataclass_fields__:
            assert field_name in d, f"Missing field: {field_name}"
