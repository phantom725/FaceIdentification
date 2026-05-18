from face_attendance import db


def test_people_and_attendance_flow(tmp_path):
    database = tmp_path / "test.sqlite3"
    db.init_db(database)

    person_id = db.add_person(database, "2026001", "张三", "计科一班", "")
    db.add_face_sample(database, person_id, [0.1, 0.2, 0.3], "sample.jpg")
    db.add_attendance_log(database, person_id, "capture.jpg", 0.88, "success", "签到成功。")

    people = db.list_people(database)
    assert len(people) == 1
    assert people[0]["face_count"] == 1

    logs = db.list_attendance(database)
    assert len(logs) == 1
    assert logs[0]["name"] == "张三"
    assert db.has_success_today(database, person_id) is True


def test_clear_attendance_logs(tmp_path):
    database = tmp_path / "test.sqlite3"
    db.init_db(database)
    person_id = db.add_person(database, "2026001", "张三")
    db.add_attendance_log(database, person_id, "capture.jpg", 0.88, "success", "签到成功。")

    deleted_count = db.clear_attendance_logs(database)

    assert deleted_count == 1
    assert db.list_attendance(database) == []


def test_delete_person_removes_samples_and_related_logs(tmp_path):
    database = tmp_path / "test.sqlite3"
    db.init_db(database)
    person_id = db.add_person(database, "2026001", "张三")
    db.add_face_sample(database, person_id, [0.1, 0.2, 0.3], "sample.jpg")
    db.add_attendance_log(database, person_id, "capture.jpg", 0.88, "success", "签到成功。")

    assert db.delete_person(database, person_id) is True

    assert db.get_person(database, person_id) is None
    assert db.list_face_embeddings(database) == []
    assert db.list_attendance(database) == []
