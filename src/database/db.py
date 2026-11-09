import sqlite3
from pathlib import Path
from typing import Dict, Optional

DB_PATH = Path(__file__).resolve().parent / "tutorflow.db"


def get_connection() -> sqlite3.Connection:
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  return conn


def init_db() -> None:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                grade INTEGER NOT NULL,
                level TEXT DEFAULT 'средний',
                weak_topics TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            )
        """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                topic TEXT NOT NULL,
                plan_text TEXT NOT NULL,
                math_solution TEXT,
                validation_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                topic TEXT NOT NULL,
                task_text TEXT NOT NULL,
                student_solution TEXT,
                verdict TEXT,
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)
    conn.commit()


def get_student(name: str) -> Optional[Dict]:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE name = ?", (name,))
    row = cursor.fetchone()
    return dict(row) if row else None


def add_student(
    name: str, grade: int, level: str = "средний", weak_topics: str = ""
) -> int:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT OR IGNORE INTO students (name, grade, level, weak_topics)
            VALUES (?, ?, ?, ?)
        """,
        (name, grade, level, weak_topics),
    )
    conn.commit()
    return cursor.lastrowid


def save_lesson(
    student_id: int,
    topic: str,
    plan_text: str,
    math_solution: str,
    validation_status: str,
) -> int:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO lessons (student_id, topic, plan_text, math_solution, validation_status)
            VALUES (?, ?, ?, ?, ?)
        """,
        (student_id, topic, plan_text, math_solution, validation_status),
    )
    conn.commit()
    return cursor.lastrowid


def save_assignment_result(
    student_id: int,
    topic: str,
    task_text: str,
    student_solution: str,
    verdict: str,
    is_correct: bool,
) -> int:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO assignments (student_id, topic, task_text, student_solution, verdict, is_correct)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            student_id,
            topic,
            task_text,
            student_solution,
            verdict,
            1 if is_correct else 0,
        ),
    )
    conn.commit()
    return cursor.lastrowid


if __name__ == "__main__":
  init_db()
  add_student(
      "Иван", grade=7, level="базовый", weak_topics="Линейные уравнения, Дроби"
  )
  print("База данных обновлена и готова:", DB_PATH)