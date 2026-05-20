from __future__ import annotations

import base64
import csv
import io
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
from flask import Flask, Response, current_app, jsonify, render_template, request

from . import db
from .face_engine import FaceDetectionError, FaceEngine, ModelMissingError


DEFAULT_CONFIG = {
    "DATABASE": "instance/attendance.sqlite3",
    "UPLOAD_DIR": "instance/uploads",
    "DETECTOR_MODEL": "models/face_detection_yunet_2023mar.onnx",
    "RECOGNIZER_MODEL": "models/face_recognition_sface_2021dec.onnx",
    "SIMILARITY_THRESHOLD": 0.363,
}


def create_app(test_config: dict | None = None) -> Flask:
    project_root = Path(__file__).resolve().parents[1]
    app = Flask(
        __name__,
        instance_relative_config=False,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.from_mapping(DEFAULT_CONFIG)
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    db.init_db(app.config["DATABASE"])

    @lru_cache(maxsize=1)
    def engine() -> FaceEngine:
        return FaceEngine(app.config["DETECTOR_MODEL"], app.config["RECOGNIZER_MODEL"])

    def model_status() -> dict:
        detector_exists = Path(app.config["DETECTOR_MODEL"]).exists()
        recognizer_exists = Path(app.config["RECOGNIZER_MODEL"]).exists()
        return {
            "ready": detector_exists and recognizer_exists,
            "detector_exists": detector_exists,
            "recognizer_exists": recognizer_exists,
        }

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            stats=db.dashboard_stats(app.config["DATABASE"]),
            model_status=model_status(),
        )

    @app.get("/people")
    def people_page():
        return render_template(
            "people.html",
            people=db.list_people(app.config["DATABASE"]),
            model_status=model_status(),
        )

    @app.get("/register")
    def register_page():
        return render_template(
            "register.html",
            people=db.list_people(app.config["DATABASE"]),
            model_status=model_status(),
        )

    @app.get("/recognize")
    def recognize_page():
        return render_template(
            "recognize.html",
            model_status=model_status(),
            threshold=app.config["SIMILARITY_THRESHOLD"],
        )

    @app.get("/attendance")
    def attendance_page():
        person_id = request.args.get("person_id", type=int)
        day = request.args.get("day") or None
        return render_template(
            "attendance.html",
            people=db.list_people(app.config["DATABASE"]),
            logs=db.list_attendance(app.config["DATABASE"], person_id=person_id, day=day),
            selected_person_id=person_id,
            selected_day=day or "",
        )

    @app.post("/api/people")
    def api_people():
        payload = request.get_json(silent=True) or request.form
        student_no = (payload.get("student_no") or "").strip()
        name = (payload.get("name") or "").strip()
        class_name = (payload.get("class_name") or "").strip()
        notes = (payload.get("notes") or "").strip()
        if not student_no or not name:
            return jsonify({"ok": False, "error": "学号和姓名不能为空。"}), 400
        try:
            person_id = db.add_person(app.config["DATABASE"], student_no, name, class_name, notes)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"新增失败：{exc}"}), 400
        return jsonify({"ok": True, "person_id": person_id})

    @app.post("/api/people/<int:person_id>/delete")
    def api_delete_person(person_id: int):
        deleted = db.delete_person(app.config["DATABASE"], person_id)
        if not deleted:
            return jsonify({"ok": False, "error": "人员不存在或已被删除。"}), 404
        return jsonify({"ok": True, "message": "人员及其人脸样本已删除。"})

    @app.post("/api/faces/register")
    def api_register_face():
        person_id = request.form.get("person_id", type=int)
        if not person_id or db.get_person(app.config["DATABASE"], person_id) is None:
            return jsonify({"ok": False, "error": "请选择有效人员"}), 400
        try:
            image_bytes = read_image_bytes()
            face_engine = engine()
            image = face_engine.decode_image(image_bytes)
            embedding = face_engine.extract_embedding(image, require_single_face=True)
            image_path = save_image_bytes(image_bytes, "register")
            sample_id = db.add_face_sample(app.config["DATABASE"], person_id, embedding, image_path)
        except (ModelMissingError, FaceDetectionError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "sample_id": sample_id, "message": "人脸样本录入成功。"})

    @app.post("/api/recognize")
    def api_recognize():
        image_path = ""
        try:
            image_bytes = read_image_bytes()
            image_path = save_image_bytes(image_bytes, "recognize")
            face_engine = engine()
            image = face_engine.decode_image(image_bytes)
            embedding = face_engine.extract_embedding(image, require_single_face=True)
            candidates = db.list_face_embeddings(app.config["DATABASE"])
            match = face_engine.find_best_match(embedding, candidates)
        except (ModelMissingError, FaceDetectionError) as exc:
            if image_path:
                db.add_attendance_log(app.config["DATABASE"], None, image_path, None, "error", str(exc))
            return jsonify({"ok": False, "error": str(exc)}), 400

        threshold = float(app.config["SIMILARITY_THRESHOLD"])
        if match is None or match.similarity < threshold:
            similarity = match.similarity if match else None
            message = "未匹配到已录入人员。"
            db.add_attendance_log(app.config["DATABASE"], None, image_path, similarity, "unknown", message)
            return jsonify(
                {
                    "ok": True,
                    "matched": False,
                    "status": "unknown",
                    "similarity": similarity,
                    "message": message,
                }
            )

        already_signed = db.has_success_today(app.config["DATABASE"], match.person_id)
        status = "duplicate" if already_signed else "success"
        message = "今日已签到，本次记录为重复识别。" if already_signed else "签到成功。"
        log_id = db.add_attendance_log(
            app.config["DATABASE"],
            match.person_id,
            image_path,
            match.similarity,
            status,
            message,
        )
        return jsonify(
            {
                "ok": True,
                "matched": True,
                "status": status,
                "message": message,
                "log_id": log_id,
                "person": {
                    "id": match.person_id,
                    "student_no": match.student_no,
                    "name": match.name,
                    "class_name": match.class_name,
                },
                "similarity": match.similarity,
                "threshold": threshold,
            }
        )

    @app.get("/api/attendance/export")
    def export_attendance():
        person_id = request.args.get("person_id", type=int)
        day = request.args.get("day") or None
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["姓名", "学号", "班级", "签到时间", "状态", "相似度", "说明"])
        for row in db.list_attendance(app.config["DATABASE"], person_id=person_id, day=day):
            writer.writerow(
                [
                    row["name"] or "未知人员",
                    row["student_no"] or "",
                    row["class_name"] or "",
                    row["checked_in_at"],
                    row["status"],
                    "" if row["similarity"] is None else f"{row['similarity']:.4f}",
                    row["message"],
                ]
            )
        csv_text = "\ufeff" + output.getvalue()
        return Response(
            csv_text,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=attendance.csv"},
        )

    @app.post("/api/attendance/clear")
    def clear_attendance():
        deleted_count = db.clear_attendance_logs(app.config["DATABASE"])
        return jsonify({"ok": True, "deleted_count": deleted_count, "message": "考勤记录已清空。"})

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "model_status": model_status()})

    return app


def read_image_bytes() -> bytes:
    file = request.files.get("image")
    if file and file.filename:
        return file.read()

    image_data = request.form.get("image_data") or (request.get_json(silent=True) or {}).get("image_data")
    if image_data:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        return base64.b64decode(image_data)
    raise FaceDetectionError("请上传图片或使用摄像头拍照。")


def save_image_bytes(image_bytes: bytes, category: str) -> str:
    upload_root = Path(current_app.config["UPLOAD_DIR"])
    target_dir = upload_root / category
    target_dir.mkdir(parents=True, exist_ok=True)
    image = FaceEngine.decode_image(image_bytes)
    filename = f"{uuid.uuid4().hex}.jpg"
    target = target_dir / filename
    cv2.imwrite(str(target), image)
    return str(target)
