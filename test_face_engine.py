import numpy as np

from face_attendance.face_engine import FaceEngine


def test_usable_faces_filters_tiny_false_positive(monkeypatch):
    """测试 usable_faces 方法能正确过滤掉面积过小的虚假人脸（假阳性）。
    
    场景：检测器返回两个框，一个是 40x40 的极小框（仅占全图约 0.17%），
    另一个是 300x360 的正常人脸框（占全图约 11.7%）。
    在 min_face_area_ratio=0.01（即 1%）的阈值下，只有大框应该被保留。
    """
    # 使用 __new__ 绕过 __init__，避免加载真实的 ONNX 模型文件
    engine = FaceEngine.__new__(FaceEngine)
    engine.min_face_area_ratio = 0.01  # 设置最小人脸面积比例为 1%

    # 模拟检测器输出的两个人脸框，每行 15 个数值：
    # [x, y, width, height, 右眼x, 右眼y, 左眼x, 左眼y, 鼻尖x, 鼻尖y,
    #  右嘴角x, 右嘴角y, 左嘴角x, 左嘴角y, 置信度]
    faces = np.array(
        [
            [10, 10, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],   # 小框：40*40=1600
            [100, 80, 300, 360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.84], # 大框：300*360=108000
        ],
        dtype=np.float32,
    )
    # 用 monkeypatch 将 detect_faces 替换为返回固定数组的 lambda，实现单元测试隔离
    monkeypatch.setattr(engine, "detect_faces", lambda image: faces)

    # 输入一张 720p 的黑图（实际像素值不影响结果，因为 detect_faces 已被 mock）
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    usable = engine.usable_faces(image)

    # 断言：只有 1 个人脸满足面积比例要求
    assert len(usable) == 1
    # 断言：保留下来的人脸宽度为 300，即第二个大框
    assert usable[0][2] == 300


def test_primary_face_box_returns_main_face_metrics(monkeypatch):
    """测试 primary_face_box 方法返回最大人脸的完整几何与关键点信息。
    
    验证内容包括：边界框坐标、中心点计算、图像尺寸、面积比例、
    关键点提取以及 yaw_proxy（偏航角近似值）的计算正确性。
    """
    engine = FaceEngine.__new__(FaceEngine)
    engine.min_face_area_ratio = 0.01

    # 构造一个带完整 5 点关键点的人脸框：
    # 眼睛1(40,60)、眼睛2(80,60)、鼻尖(64,90)、嘴1(45,120)、嘴2(82,120)
    faces = np.array(
        [
            [10, 20, 100, 120, 40, 60, 80, 60, 64, 90, 45, 120, 82, 120, 0.91],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(engine, "detect_faces", lambda image: faces)

    # 输入 400x500 的图像
    face_box = engine.primary_face_box(np.zeros((400, 500, 3), dtype=np.uint8))

    # 验证边界框左上角 x 坐标
    assert face_box["x"] == 10
    # 验证中心点 x：10 + 100/2 = 60
    assert face_box["center_x"] == 60
    # 验证图像宽度被正确记录
    assert face_box["frame_width"] == 500
    # 验证面积比例：100*120 / (400*500) = 12000/200000 = 0.06
    assert np.isclose(face_box["area_ratio"], 0.06)
    # 验证鼻尖关键点 x 坐标被正确提取
    assert face_box["landmarks"]["nose"]["x"] == 64
    # 验证 yaw_proxy 计算：
    # eye_center_x = (40+80)/2 = 60, mouth_center_x = (45+82)/2 = 63.5
    # landmark_center_x = (60+63.5)/2 = 61.75
    # yaw_proxy = (64 - 61.75) / 100 = 0.0225
    assert np.isclose(face_box["yaw_proxy"], 0.0225)


def test_anti_spoof_uses_logits_to_classify_real(monkeypatch):
    """测试 anti_spoof 活体检测方法能根据模型 logits 正确判定为真人。
    
    使用 FakeNet 模拟 ONNX 反欺骗网络，固定返回 real_logit=2.0、spoof_logit=-1.0。
    在 threshold=0.0 的设置下，差值 3.0 > 0，应判定为真人。
    同时验证输入 blob 的预处理形状为 (1, 3, 128, 128)。
    """
    # 定义假的神经网络对象，模拟 cv2.dnn.Net 的 setInput 和 forward 接口
    class FakeNet:
        def setInput(self, blob):
            # 保存传入的预处理 blob，供后续断言检查
            self.blob = blob

        def forward(self):
            # 返回固定 logits：[[真人分数, 欺骗分数]]
            return np.array([[2.0, -1.0]], dtype=np.float32)

    engine = FaceEngine.__new__(FaceEngine)
    engine.min_face_area_ratio = 0.01
    engine.anti_spoof_net = FakeNet()          # 注入假网络
    engine.anti_spoof_threshold = 0.0          # 差值 >= 0 即判定为真人
    engine.anti_spoof_input_size = 128         # 模型输入尺寸 128x128
    engine.anti_spoof_crop_expansion = 1.5   # 裁剪扩展倍数

    # 构造一个 80x80 的人脸框，确保裁剪后经过预处理能得到 128x128 的 blob
    faces = np.array(
        [
            [30, 30, 80, 80, 45, 55, 95, 55, 70, 80, 50, 100, 90, 100, 0.92],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(engine, "detect_faces", lambda image: faces)

    # 执行活体检测
    result = engine.anti_spoof(np.zeros((180, 180, 3), dtype=np.uint8))

    # 断言判定结果为真人
    assert result.is_real is True
    assert result.status == "real"
    # 断言 logit 差值：2.0 - (-1.0) = 3.0
    assert result.logit_diff == 3.0
    # 断言传入 FakeNet 的 blob 形状符合 NCHW 格式：(batch, channel, height, width)
    assert engine.anti_spoof_net.blob.shape == (1, 3, 128, 128)


def test_anti_spoof_blob_converts_bgr_to_rgb():
    """测试 _anti_spoof_blob 私有方法在预处理时正确将 BGR 转换为 RGB。
    
    构造一张纯蓝色图像（OpenCV BGR 格式下 channel 0 = 255）。
    预处理后，在 RGB 格式下该颜色应出现在 channel 2（红色通道），
    而 channel 0（蓝色通道）应为 0。
    同时验证归一化到 [0, 1] 的结果。
    """
    engine = FaceEngine.__new__(FaceEngine)
    engine.anti_spoof_input_size = 128

    # 创建 20x20 的纯蓝色图像：BGR 中 B=255, G=0, R=0
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[:, :, 0] = 255

    blob = engine._anti_spoof_blob(crop)

    # blob 形状为 (1, 3, 128, 128)，已归一化到 [0, 1]
    # 中心点坐标 (64, 64) 在填充后的正方形中心附近
    # 原图纯蓝色 BGR(255,0,0) -> RGB(0,0,255)，所以：
    # channel 2（R）应为 255/255 = 1.0
    assert blob[0, 2, 64, 64] == 1.0
    # channel 0（B）应为 0/255 = 0.0
    assert blob[0, 0, 64, 64] == 0.0