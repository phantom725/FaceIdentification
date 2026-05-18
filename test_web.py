from face_attendance.web import create_app


def make_app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "test.sqlite3"),
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "DETECTOR_MODEL": str(tmp_path / "missing_detector.onnx"),
            "RECOGNIZER_MODEL": str(tmp_path / "missing_recognizer.onnx"),
        }
    )


def test_add_person_and_export(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/people",
        data={"student_no": "2026001", "name": "张三", "class_name": "计科一班"},
    )
    assert response.status_code == 200
    assert response.get_json()["ok"] is True

    export = client.get("/api/attendance/export")
    assert export.status_code == 200
    assert "姓名" in export.get_data(as_text=True)


def test_clear_attendance_endpoint(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    client.post("/api/people", data={"student_no": "2026001", "name": "张三"})
    clear_response = client.post("/api/attendance/clear")

    assert clear_response.status_code == 200
    assert clear_response.get_json()["ok"] is True


def test_delete_person_endpoint(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    create_response = client.post("/api/people", data={"student_no": "2026001", "name": "张三"})
    person_id = create_response.get_json()["person_id"]
    delete_response = client.post(f"/api/people/{person_id}/delete")

    assert delete_response.status_code == 200
    assert delete_response.get_json()["ok"] is True


def test_pages_render(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    for path in ["/", "/people", "/register", "/recognize", "/attendance"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "人脸识别考勤系统" in response.get_data(as_text=True)


def test_missing_model_returns_clear_error(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    client.post("/api/people", data={"student_no": "2026001", "name": "张三"})

    response = client.post("/api/faces/register", data={"person_id": "1"})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert "请上传图片" in payload["error"]
