from __future__ import annotations

from pathlib import Path

from face_attendance import db


def main() -> None:
    database = Path("instance/attendance.sqlite3")
    db.init_db(database)
    demo_people = [
        ("2026001", "张三", "计科一班", "演示账号"),
        ("2026002", "李四", "计科一班", "演示账号"),
    ]
    for student_no, name, class_name, notes in demo_people:
        try:
            db.add_person(database, student_no, name, class_name, notes)
            print(f"Added {student_no} {name}")
        except Exception:
            print(f"Skip {student_no}: already exists")


if __name__ == "__main__":
    main()

