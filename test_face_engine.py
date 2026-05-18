import numpy as np

from face_attendance.face_engine import FaceEngine


def test_usable_faces_filters_tiny_false_positive(monkeypatch):
    engine = FaceEngine.__new__(FaceEngine)
    engine.min_face_area_ratio = 0.01

    faces = np.array(
        [
            [10, 10, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],
            [100, 80, 300, 360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.84],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(engine, "detect_faces", lambda image: faces)

    usable = engine.usable_faces(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert len(usable) == 1
    assert usable[0][2] == 300


def test_primary_face_box_returns_main_face_metrics(monkeypatch):
    engine = FaceEngine.__new__(FaceEngine)
    engine.min_face_area_ratio = 0.01

    faces = np.array(
        [
            [10, 20, 100, 120, 40, 60, 80, 60, 64, 90, 45, 120, 82, 120, 0.91],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(engine, "detect_faces", lambda image: faces)

    face_box = engine.primary_face_box(np.zeros((400, 500, 3), dtype=np.uint8))

    assert face_box["x"] == 10
    assert face_box["center_x"] == 60
    assert face_box["frame_width"] == 500
    assert np.isclose(face_box["area_ratio"], 0.06)
    assert face_box["landmarks"]["nose"]["x"] == 64
    assert np.isclose(face_box["yaw_proxy"], 0.0225)


def test_anti_spoof_uses_logits_to_classify_real(monkeypatch):
    class FakeNet:
        def setInput(self, blob):
            self.blob = blob

        def forward(self):
            return np.array([[2.0, -1.0]], dtype=np.float32)

    engine = FaceEngine.__new__(FaceEngine)
    engine.min_face_area_ratio = 0.01
    engine.anti_spoof_net = FakeNet()
    engine.anti_spoof_threshold = 0.0
    engine.anti_spoof_input_size = 128
    engine.anti_spoof_crop_expansion = 1.5
    faces = np.array(
        [
            [30, 30, 80, 80, 45, 55, 95, 55, 70, 80, 50, 100, 90, 100, 0.92],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(engine, "detect_faces", lambda image: faces)

    result = engine.anti_spoof(np.zeros((180, 180, 3), dtype=np.uint8))

    assert result.is_real is True
    assert result.status == "real"
    assert result.logit_diff == 3.0
    assert engine.anti_spoof_net.blob.shape == (1, 3, 128, 128)


def test_anti_spoof_blob_converts_bgr_to_rgb():
    engine = FaceEngine.__new__(FaceEngine)
    engine.anti_spoof_input_size = 128
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[:, :, 0] = 255

    blob = engine._anti_spoof_blob(crop)

    assert blob[0, 2, 64, 64] == 1.0
    assert blob[0, 0, 64, 64] == 0.0
