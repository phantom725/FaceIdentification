from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


class ModelMissingError(RuntimeError):
    """Raised when the ONNX model files have not been downloaded yet."""


class FaceDetectionError(RuntimeError):
    """Raised when a registration or recognition image is not usable."""


class NoFaceDetectedError(FaceDetectionError):
    """Raised when an image is valid but does not contain a usable face."""


@dataclass
class AntiSpoofResult:
    is_real: bool
    status: str
    confidence: float
    real_logit: float
    spoof_logit: float
    logit_diff: float


@dataclass
class RecognitionMatch:
    person_id: int
    sample_id: int
    student_no: str
    name: str
    class_name: str
    similarity: float


class FaceEngine:
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
        self.detector_model = Path(detector_model)
        self.recognizer_model = Path(recognizer_model)
        self.anti_spoof_model = Path(anti_spoof_model) if anti_spoof_model else None
        if not self.detector_model.exists() or not self.recognizer_model.exists():
            raise ModelMissingError(
                "ONNX 模型不存在，请先运行 python scripts/download_models.py，"
                "或将 YuNet/SFace 模型手动放入 models/ 目录。"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(self.detector_model),
            "",
            (320, 320),
            detector_score_threshold,
            nms_threshold,
            top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(self.recognizer_model), "")
        self.min_face_area_ratio = min_face_area_ratio
        self.anti_spoof_net = None
        if self.anti_spoof_model and self.anti_spoof_model.exists():
            self.anti_spoof_net = cv2.dnn.readNetFromONNX(str(self.anti_spoof_model))
        self.anti_spoof_threshold = anti_spoof_threshold
        self.anti_spoof_input_size = anti_spoof_input_size
        self.anti_spoof_crop_expansion = anti_spoof_crop_expansion

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        try:
            pil_image = Image.open(BytesIO(image_bytes))
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            raise FaceDetectionError("无法解析图片，请上传 jpg、jpeg 或 png 格式图片。")
        return image

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_vec = left.astype(np.float32).reshape(-1)
        right_vec = right.astype(np.float32).reshape(-1)
        left_norm = np.linalg.norm(left_vec)
        right_norm = np.linalg.norm(right_vec)
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))

    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces

    def usable_faces(self, image: np.ndarray) -> list[np.ndarray]:
        height, width = image.shape[:2]
        image_area = float(width * height)
        faces = self.detect_faces(image)
        usable = [
            face
            for face in faces
            if float(face[2] * face[3]) / image_area >= self.min_face_area_ratio
        ]
        return sorted(usable, key=lambda face: float(face[2] * face[3]), reverse=True)

    def primary_face_box(self, image: np.ndarray) -> dict[str, float]:
        faces = self.usable_faces(image)
        if len(faces) == 0:
            raise NoFaceDetectedError("未检测到人脸，请正对摄像头并保持光线清晰。")

        height, width = image.shape[:2]
        face = faces[0]
        x = float(face[0])
        y = float(face[1])
        box_width = float(face[2])
        box_height = float(face[3])
        area_ratio = (box_width * box_height) / float(width * height)
        score = float(face[14]) if len(face) > 14 else 0.0
        landmarks = {
            "eye_1": {"x": float(face[4]), "y": float(face[5])},
            "eye_2": {"x": float(face[6]), "y": float(face[7])},
            "nose": {"x": float(face[8]), "y": float(face[9])},
            "mouth_1": {"x": float(face[10]), "y": float(face[11])},
            "mouth_2": {"x": float(face[12]), "y": float(face[13])},
        }
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
        if self.anti_spoof_net is None:
            raise ModelMissingError("反欺骗 ONNX 模型不存在，请先运行 python scripts/download_models.py。")
        faces = self.usable_faces(image)
        if len(faces) == 0:
            raise NoFaceDetectedError("未检测到人脸，请正对摄像头并保持光线清晰。")

        crop = self._expanded_face_crop(image, faces[0])
        blob = self._anti_spoof_blob(crop)
        self.anti_spoof_net.setInput(blob)
        logits = self.anti_spoof_net.forward().reshape(-1)
        if logits.size != 2:
            raise FaceDetectionError("反欺骗模型输出异常，无法判断真人或照片。")

        real_logit = float(logits[0])
        spoof_logit = float(logits[1])
        logit_diff = real_logit - spoof_logit
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
        image_height, image_width = image.shape[:2]
        x, y, width, height = [float(value) for value in face[:4]]
        max_dim = max(width, height)
        center_x = x + width / 2
        center_y = y + height / 2
        crop_size = int(max_dim * self.anti_spoof_crop_expansion)
        if crop_size <= 0:
            raise FaceDetectionError("人脸区域无效，无法进行反欺骗检测。")

        left = int(center_x - crop_size / 2)
        top = int(center_y - crop_size / 2)
        right = left + crop_size
        bottom = top + crop_size
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(image_width, right)
        crop_bottom = min(image_height, bottom)
        if crop_right <= crop_left or crop_bottom <= crop_top:
            raise FaceDetectionError("人脸区域无效，无法进行反欺骗检测。")

        crop = image[crop_top:crop_bottom, crop_left:crop_right]
        top_pad = max(0, -top)
        left_pad = max(0, -left)
        bottom_pad = max(0, bottom - image_height)
        right_pad = max(0, right - image_width)
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
        target_size = self.anti_spoof_input_size
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        height, width = crop.shape[:2]
        ratio = float(target_size) / max(height, width)
        resized_width = int(width * ratio)
        resized_height = int(height * ratio)
        interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
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
        return padded.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255.0

    def extract_embedding(self, image: np.ndarray, require_single_face: bool = True) -> list[float]:
        faces = self.usable_faces(image)
        if len(faces) == 0:
            raise NoFaceDetectedError("未检测到人脸，请换一张正脸、光线更清晰的图片。")
        if require_single_face and len(faces) > 1:
            main_area = float(faces[0][2] * faces[0][3])
            second_area = float(faces[1][2] * faces[1][3])
            if second_area / main_area > 0.55:
                raise FaceDetectionError("检测到多张较明显的人脸，请上传只包含一人的图片。")

        face = faces[0]
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned)
        vector = feature.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def find_best_match(
        self,
        embedding: list[float],
        candidates: list[dict],
    ) -> RecognitionMatch | None:
        if not candidates:
            return None
        query = np.array(embedding, dtype=np.float32)
        best: RecognitionMatch | None = None
        for item in candidates:
            similarity = self.cosine_similarity(query, np.array(item["embedding"], dtype=np.float32))
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
