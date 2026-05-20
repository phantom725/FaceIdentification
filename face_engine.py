from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


class ModelMissingError(RuntimeError):
    """当 ONNX 模型文件尚未下载时抛出此异常。"""


class FaceDetectionError(RuntimeError):
    """当注册或识别图片无法使用时抛出此异常。"""


class NoFaceDetectedError(FaceDetectionError):
    """当图片有效但未检测到可用的人脸时抛出此异常。"""


@dataclass
class AntiSpoofResult:
    """活体检测（反欺骗）的结果数据类。
    
    Attributes:
        is_real: 是否为真人（True）或照片/屏幕（False）。
        status: 检测状态字符串，"real" 或 "spoof"。
        confidence: 置信度，取两个 logits 差值的绝对值。
        real_logit: 真人分类的 logit 值。
        spoof_logit: 欺骗分类的 logit 值。
        logit_diff: 真人 logit 与欺骗 logit 的差值。
    """
    is_real: bool
    status: str
    confidence: float
    real_logit: float
    spoof_logit: float
    logit_diff: float


@dataclass
class RecognitionMatch:
    """人脸识别匹配结果的数据类。
    
    Attributes:
        person_id: 人员唯一标识 ID。
        sample_id: 样本 ID。
        student_no: 学号。
        name: 姓名。
        class_name: 班级名称。
        similarity: 与查询人脸的余弦相似度（范围 0~1）。
    """
    person_id: int
    sample_id: int
    student_no: str
    name: str
    class_name: str
    similarity: float


class FaceEngine:
    """人脸引擎，封装了人脸检测、特征提取、活体检测和 1:N 匹配功能。
    
    基于 OpenCV DNN 模块的 YuNet（检测器）和 SFace（识别器）ONNX 模型实现。
    可选支持反欺骗（anti-spoof）ONNX 模型以区分真人与照片攻击。
    """

    def __init__(
        self,
        detector_model: str | Path,
        recognizer_model: str | Path,
        anti_spoof_model: str | Path | None = None,
        detector_score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        min_face_area_ratio: float = 0.01,
        anti_spoof_threshold: float = 0.0,
        anti_spoof_input_size: int = 128,
        anti_spoof_crop_expansion: float = 1.5,
    ) -> None:
        """初始化人脸引擎。
        
        Args:
            detector_model: YuNet 人脸检测器 ONNX 模型路径。
            recognizer_model: SFace 人脸识别器 ONNX 模型路径。
            anti_spoof_model: 反欺骗 ONNX 模型路径（可选）。
            detector_score_threshold: 检测置信度阈值，低于此值的检测框会被过滤。
            nms_threshold: 非极大值抑制（NMS）的 IoU 阈值。
            top_k: 检测器保留的最大候选框数量。
            min_face_area_ratio: 最小人脸面积占整图面积的比例，用于过滤过小的人脸。
            anti_spoof_threshold: 活体检测阈值，logit 差值大于等于此值判定为真人。
            anti_spoof_input_size: 反欺骗模型输入图像的边长（正方形）。
            anti_spoof_crop_expansion: 反欺骗裁剪时相对于人脸框的扩展倍数。
        
        Raises:
            ModelMissingError: 当检测器或识别器模型文件不存在时抛出。
        """
        self.detector_model = Path(detector_model)
        self.recognizer_model = Path(recognizer_model)
        self.anti_spoof_model = Path(anti_spoof_model) if anti_spoof_model else None
        
        # 检查核心模型文件是否存在
        if not self.detector_model.exists() or not self.recognizer_model.exists():
            raise ModelMissingError(
                "ONNX 模型不存在，请先运行 python scripts/download_models.py，"
                "或将 YuNet/SFace 模型手动放入 models/ 目录。"
            )

        # 初始化 YuNet 人脸检测器
        self.detector = cv2.FaceDetectorYN.create(
            str(self.detector_model),
            "",
            (320, 320),
            detector_score_threshold,
            nms_threshold,
            top_k,
        )
        
        # 初始化 SFace 人脸识别器
        self.recognizer = cv2.FaceRecognizerSF.create(str(self.recognizer_model), "")
        
        self.min_face_area_ratio = min_face_area_ratio
        
        # 可选：加载反欺骗模型
        self.anti_spoof_net = None
        if self.anti_spoof_model and self.anti_spoof_model.exists():
            self.anti_spoof_net = cv2.dnn.readNetFromONNX(str(self.anti_spoof_model))
        
        self.anti_spoof_threshold = anti_spoof_threshold
        self.anti_spoof_input_size = anti_spoof_input_size
        self.anti_spoof_crop_expansion = anti_spoof_crop_expansion

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """将图片字节流解码为 OpenCV 格式的 BGR 图像数组。
        
        支持自动处理 EXIF 方向信息，并统一转换为 RGB 后再转 BGR。
        
        Args:
            image_bytes: 图片文件的二进制数据。
        
        Returns:
            解码后的 BGR 格式 numpy 数组。
        
        Raises:
            FaceDetectionError: 当图片格式无法解析时抛出。
        """
        try:
            pil_image = Image.open(BytesIO(image_bytes))
            # 根据 EXIF 信息自动旋转图片，并统一转换为 RGB
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            # PIL 的 RGB 转为 numpy 后，再转为 OpenCV 的 BGR 格式
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            raise FaceDetectionError("无法解析图片，请上传 jpg、jpeg 或 png 格式图片。")
        return image

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        """计算两个向量之间的余弦相似度。
        
        结果范围在 [-1, 1] 之间，对于已归一化的人脸特征向量通常在 [0, 1] 之间。
        
        Args:
            left: 左侧向量。
            right: 右侧向量。
        
        Returns:
            两个向量的余弦相似度；若任一向量为零向量则返回 0.0。
        """
        left_vec = left.astype(np.float32).reshape(-1)
        right_vec = right.astype(np.float32).reshape(-1)
        left_norm = np.linalg.norm(left_vec)
        right_norm = np.linalg.norm(right_vec)
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))

    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        """在整张图像上运行人脸检测。
        
        Args:
            image: BGR 格式的输入图像。
        
        Returns:
            检测到的所有人脸框数组，形状为 (N, 15)。
            每行包含：x, y, w, h, 右眼x, 右眼y, 左眼x, 左眼y, 
            鼻尖x, 鼻尖y, 右嘴角x, 右嘴角y, 左嘴角x, 左嘴角y, 置信度。
            若未检测到人脸则返回空数组。
        """
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces

    def usable_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """检测图像中的人脸，并按面积过滤掉过小的人脸，返回可用的人脸列表。
        
        Args:
            image: BGR 格式的输入图像。
        
        Returns:
            按人脸面积从大到小排序的可用人脸框列表。
        """
        height, width = image.shape[:2]
        image_area = float(width * height)
        faces = self.detect_faces(image)
        # 过滤掉面积小于设定比例的人脸，避免远处模糊人脸干扰
        usable = [
            face
            for face in faces
            if float(face[2] * face[3]) / image_area >= self.min_face_area_ratio
        ]
        # 按面积降序排列，最大的排在最前
        return sorted(usable, key=lambda face: float(face[2] * face[3]), reverse=True)

    def primary_face_box(self, image: np.ndarray) -> dict[str, float]:
        """获取图像中最大可用人脸的详细信息。
        
        除了返回边界框外，还计算了人脸中心点、面积占比、5 个关键点坐标
        以及一个基于鼻尖偏移的 yaw_proxy（偏航角近似值）。
        
        Args:
            image: BGR 格式的输入图像。
        
        Returns:
            包含人脸位置、尺寸、关键点及姿态近似信息的字典。
        
        Raises:
            NoFaceDetectedError: 当未检测到可用的人脸时抛出。
        """
        faces = self.usable_faces(image)
        if len(faces) == 0:
            raise NoFaceDetectedError("未检测到人脸，请正对摄像头并保持光线清晰。")

        height, width = image.shape[:2]
        face = faces[0]  # 取最大的人脸
        
        x = float(face[0])
        y = float(face[1])
        box_width = float(face[2])
        box_height = float(face[3])
        area_ratio = (box_width * box_height) / float(width * height)
        score = float(face[14]) if len(face) > 14 else 0.0
        
        # 提取 5 个面部关键点（双眼、鼻尖、双嘴角）
        landmarks = {
            "eye_1": {"x": float(face[4]), "y": float(face[5])},
            "eye_2": {"x": float(face[6]), "y": float(face[7])},
            "nose": {"x": float(face[8]), "y": float(face[9])},
            "mouth_1": {"x": float(face[10]), "y": float(face[11])},
            "mouth_2": {"x": float(face[12]), "y": float(face[13])},
        }
        
        # 通过鼻尖相对于眼-嘴中心线的水平偏移，近似估算人脸偏航角（yaw）
        eye_center_x = (landmarks["eye_1"]["x"] + landmarks["eye_2"]["x"]) / 2
        mouth_center_x = (landmarks["mouth_1"]["x"] + landmarks["mouth_2"]["x"]) / 2
        landmark_center_x = (eye_center_x + mouth_center_x) / 2
        yaw_proxy = (landmarks["nose"]["x"] - landmark_center_x) / max(box_width, 1.0)
        
        return {
            "x": x,
            "y": y,
            "width": box_width,
            "height": box_height,
            "center_x": x + box_width / 2,
            "center_y": y + box_height / 2,
            "area_ratio": area_ratio,
            "frame_width": float(width),
            "frame_height": float(height),
            "score": score,
            "landmarks": landmarks,
            "yaw_proxy": yaw_proxy,
        }

    def anti_spoof(self, image: np.ndarray) -> AntiSpoofResult:
        """对图像进行活体检测（反欺骗），判断是否为真人。
        
        流程：检测人脸 -> 扩展裁剪人脸区域 -> 预处理 -> 送入 ONNX 模型推理。
        
        Args:
            image: BGR 格式的输入图像。
        
        Returns:
            AntiSpoofResult 数据类，包含真人/欺骗判定及置信度信息。
        
        Raises:
            ModelMissingError: 当反欺骗模型未加载时抛出。
            NoFaceDetectedError: 当未检测到人脸时抛出。
            FaceDetectionError: 当模型输出异常时抛出。
        """
        if self.anti_spoof_net is None:
            raise ModelMissingError("反欺骗 ONNX 模型不存在，请先运行 python scripts/download_models.py。")
        
        faces = self.usable_faces(image)
        if len(faces) == 0:
            raise NoFaceDetectedError("未检测到人脸，请正对摄像头并保持光线清晰。")

        # 对最大的人脸进行扩展裁剪，以包含更多上下文信息供反欺骗模型判断
        crop = self._expanded_face_crop(image, faces[0])
        blob = self._anti_spoof_blob(crop)
        
        self.anti_spoof_net.setInput(blob)
        logits = self.anti_spoof_net.forward().reshape(-1)
        if logits.size != 2:
            raise FaceDetectionError("反欺骗模型输出异常，无法判断真人或照片。")

        real_logit = float(logits[0])
        spoof_logit = float(logits[1])
        logit_diff = real_logit - spoof_logit
        # 差值大于阈值则判定为真人
        is_real = logit_diff >= self.anti_spoof_threshold
        
        return AntiSpoofResult(
            is_real=bool(is_real),
            status="real" if is_real else "spoof",
            confidence=abs(logit_diff),
            real_logit=real_logit,
            spoof_logit=spoof_logit,
            logit_diff=logit_diff,
        )

    def _expanded_face_crop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """以人脸框为中心进行扩展裁剪，用于反欺骗模型输入。
        
        若裁剪区域超出图像边界，使用 BORDER_REFLECT_101 方式填充，避免黑边。
        
        Args:
            image: 原始 BGR 图像。
            face: 单个人脸框数组，包含至少 x, y, w, h 信息。
        
        Returns:
            裁剪并可能填充后的图像区域。
        
        Raises:
            FaceDetectionError: 当裁剪区域计算无效时抛出。
        """
        image_height, image_width = image.shape[:2]
        x, y, width, height = [float(value) for value in face[:4]]
        
        # 以人脸框最大边为基准，按设定倍数扩展裁剪区域
        max_dim = max(width, height)
        center_x = x + width / 2
        center_y = y + height / 2
        crop_size = int(max_dim * self.anti_spoof_crop_expansion)
        if crop_size <= 0:
            raise FaceDetectionError("人脸区域无效，无法进行反欺骗检测。")

        # 计算扩展后的裁剪边界
        left = int(center_x - crop_size / 2)
        top = int(center_y - crop_size / 2)
        right = left + crop_size
        bottom = top + crop_size
        
        # 限制在图像有效范围内
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(image_width, right)
        crop_bottom = min(image_height, bottom)
        
        if crop_right <= crop_left or crop_bottom <= crop_top:
            raise FaceDetectionError("人脸区域无效，无法进行反欺骗检测。")

        crop = image[crop_top:crop_bottom, crop_left:crop_right]
        
        # 计算超出边界的填充量
        top_pad = max(0, -top)
        left_pad = max(0, -left)
        bottom_pad = max(0, bottom - image_height)
        right_pad = max(0, right - image_width)
        
        # 若存在越界，使用反射填充保持纹理连续性
        if top_pad or left_pad or bottom_pad or right_pad:
            crop = cv2.copyMakeBorder(
                crop,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_REFLECT_101,
            )
        return crop

    def _anti_spoof_blob(self, crop: np.ndarray) -> np.ndarray:
        """将裁剪后的人脸图像预处理为反欺骗模型所需的输入 blob。
        
        处理流程：BGR -> RGB -> 等比例缩放 -> 居中填充为正方形 -> 
        维度重排 (HWC -> CHW) -> 归一化到 [0, 1] -> 增加 batch 维度。
        
        Args:
            crop: 裁剪后的 BGR 图像。
        
        Returns:
            形状为 (1, 3, input_size, input_size) 的 float32 numpy 数组。
        """
        target_size = self.anti_spoof_input_size
        
        # 反欺骗模型通常使用 RGB 输入
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        height, width = crop.shape[:2]
        
        # 等比例缩放，使长边等于 target_size
        ratio = float(target_size) / max(height, width)
        resized_width = int(width * ratio)
        resized_height = int(height * ratio)
        # 放大时使用 LANCZOS4 插值以获得更好质量，缩小时使用 INTER_AREA 减少混叠
        interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
        
        # 计算填充量，使图像居中成为正方形
        delta_width = target_size - resized_width
        delta_height = target_size - resized_height
        top = delta_height // 2
        bottom = delta_height - top
        left = delta_width // 2
        right = delta_width - left
        
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )
        
        # 调整维度顺序并归一化
        return padded.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255.0

    def extract_embedding(self, image: np.ndarray, require_single_face: bool = True) -> list[float]:
        """提取人脸特征向量（embedding）。
        
        使用 SFace 识别器对齐裁剪后提取 128 维特征，并进行 L2 归一化。
        
        Args:
            image: BGR 格式的输入图像。
            require_single_face: 是否要求图像中只有一张明显的人脸。
                若为 True，当检测到多张面积相近的人脸时会报错。
        
        Returns:
            L2 归一化后的特征向量，以 Python 列表形式返回。
        
        Raises:
            NoFaceDetectedError: 当未检测到人脸时抛出。
            FaceDetectionError: 当检测到多张明显人脸且 require_single_face=True 时抛出。
        """
        faces = self.usable_faces(image)
        if len(faces) == 0:
            raise NoFaceDetectedError("未检测到人脸，请换一张正脸、光线更清晰的图片。")
        
        # 若要求单人脸，检查第二大脸是否过于明显（面积 > 最大脸的 55%）
        if require_single_face and len(faces) > 1:
            main_area = float(faces[0][2] * faces[0][3])
            second_area = float(faces[1][2] * faces[1][3])
            if second_area / main_area > 0.55:
                raise FaceDetectionError("检测到多张较明显的人脸，请上传只包含一人的图片。")

        face = faces[0]
        # 使用 SFace 的对齐裁剪功能，将人脸转正并裁剪为标准大小
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned)
        
        vector = feature.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm  # L2 归一化，使相似度计算更稳定
        return vector.tolist()

    def find_best_match(
        self,
        embedding: list[float],
        candidates: list[dict],
    ) -> RecognitionMatch | None:
        """在候选库中查找与查询特征最相似的人脸。
        
        遍历所有候选，计算余弦相似度，返回相似度最高的记录。
        
        Args:
            embedding: 查询人脸的 L2 归一化特征向量。
            candidates: 候选列表，每个元素为字典，需包含：
                person_id, sample_id, student_no, name, class_name, embedding。
        
        Returns:
            最佳匹配的 RecognitionMatch 对象；若候选列表为空则返回 None。
        """
        if not candidates:
            return None
        
        query = np.array(embedding, dtype=np.float32)
        best: RecognitionMatch | None = None
        
        for item in candidates:
            similarity = self.cosine_similarity(
                query, np.array(item["embedding"], dtype=np.float32)
            )
            if best is None or similarity > best.similarity:
                best = RecognitionMatch(
                    person_id=int(item["person_id"]),
                    sample_id=int(item["sample_id"]),
                    student_no=str(item["student_no"]),
                    name=str(item["name"]),
                    class_name=str(item["class_name"] or ""),
                    similarity=similarity,
                )
        return best