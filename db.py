from __future__ import annotations



import json
import sqlite3
from datetime import date, datetime, time
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_no TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    class_name TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    image_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    captured_image_path TEXT NOT NULL,
    similarity REAL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    checked_in_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def utc_now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_person(
    db_path: str | Path,
    student_no: str,
    name: str,
    class_name: str = "",
    notes: str = "",
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO people (student_no, name, class_name, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_no.strip(), name.strip(), class_name.strip(), notes.strip(), utc_now_text()),
        )
        return int(cur.lastrowid)


def list_people(db_path: str | Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT p.*,
                   COUNT(fs.id) AS face_count,
                   MAX(fs.created_at) AS last_face_at
            FROM people p
            LEFT JOIN face_samples fs ON fs.person_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """
        ).fetchall()


def get_person(db_path: str | Path, person_id: int) -> sqlite3.Row | None:
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()


def delete_person(db_path: str | Path, person_id: int) -> bool:
    with connect(db_path) as conn:
        person = conn.execute("SELECT id FROM people WHERE id = ?", (person_id,)).fetchone()
        if person is None:
            return False
        conn.execute("DELETE FROM attendance_logs WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        return True


def add_face_sample(
    db_path: str | Path,
    person_id: int,
    embedding: list[float],
    image_path: str,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO face_samples (person_id, embedding_json, image_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (person_id, json.dumps(embedding), image_path, utc_now_text()),
        )
        return int(cur.lastrowid)


def list_face_embeddings(db_path: str | Path) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT fs.id AS sample_id,
                   fs.embedding_json,
                   p.id AS person_id,
                   p.student_no,
                   p.name,
                   p.class_name
            FROM face_samples fs
            JOIN people p ON p.id = fs.person_id
            ORDER BY fs.created_at DESC
            """
        ).fetchall()
    return [
        {
            "sample_id": row["sample_id"],
            "person_id": row["person_id"],
            "student_no": row["student_no"],
            "name": row["name"],
            "class_name": row["class_name"],
            "embedding": json.loads(row["embedding_json"]),
        }
        for row in rows
    ]


def has_success_today(db_path: str | Path, person_id: int) -> bool:
    start = datetime.combine(date.today(), time.min).strftime("%Y-%m-%d %H:%M:%S")
    end = datetime.combine(date.today(), time.max).strftime("%Y-%m-%d %H:%M:%S")
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM attendance_logs
            WHERE person_id = ?
              AND status IN ('success', 'duplicate')
              AND checked_in_at BETWEEN ? AND ?
            LIMIT 1
            """,
            (person_id, start, end),
        ).fetchone()
        return row is not None


def add_attendance_log(
    db_path: str | Path,
    person_id: int | None,
    captured_image_path: str,
    similarity: float | None,
    status: str,
    message: str,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO attendance_logs
                (person_id, captured_image_path, similarity, status, message, checked_in_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (person_id, captured_image_path, similarity, status, message, utc_now_text()),
        )
        return int(cur.lastrowid)


def list_attendance(
    db_path: str | Path,
    person_id: int | None = None,
    day: str | None = None,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[Any] = []
    if person_id:
        clauses.append("a.person_id = ?")
        params.append(person_id)
    if day:
        clauses.append("date(a.checked_in_at) = date(?)")
        params.append(day)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(db_path) as conn:
        return conn.execute(
            f"""
            SELECT a.*,
                   p.student_no,
                   p.name,
                   p.class_name
            FROM attendance_logs a
            LEFT JOIN people p ON p.id = a.person_id
            {where_sql}
            ORDER BY a.checked_in_at DESC
            """,
            params,
        ).fetchall()


def clear_attendance_logs(db_path: str | Path) -> int:
    with connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM attendance_logs").fetchone()[0]
        conn.execute("DELETE FROM attendance_logs")
        return int(count)


def dashboard_stats(db_path: str | Path) -> dict[str, Any]:
    today = date.today().isoformat()
    with connect(db_path) as conn:
        people_count = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        face_count = conn.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0]
        today_count = conn.execute(
            """
            SELECT COUNT(DISTINCT person_id)
            FROM attendance_logs
            WHERE person_id IS NOT NULL
              AND status IN ('success', 'duplicate')
              AND date(checked_in_at) = date(?)
            """,
            (today,),
        ).fetchone()[0]
        recent = conn.execute(
            """
            SELECT a.*, p.name, p.student_no
            FROM attendance_logs a
            LEFT JOIN people p ON p.id = a.person_id
            ORDER BY a.checked_in_at DESC
            LIMIT 8
            """
        ).fetchall()
    return {
        "people_count": people_count,
        "face_count": face_count,
        "today_count": today_count,
        "recent": recent,
    }
