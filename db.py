"""대화 요약 + 질문 패턴을 SQLite에 영구 저장."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import config

DB_PATH = config.BASE_DIR / "storage" / "history.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                summary         TEXT    NOT NULL,
                topics          TEXT    NOT NULL,  -- JSON 배열 ["교육훈련", "보관기간"]
                question_pattern TEXT   NOT NULL   -- "기간 문의" 등 한 줄 패턴
            )
        """)


def save_log(session_id: str, summary: str, topics: list[str], question_pattern: str) -> None:
    init_db()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conversation_logs (session_id, timestamp, summary, topics, question_pattern) VALUES (?, ?, ?, ?, ?)",
            (session_id, datetime.now().isoformat(), summary, json.dumps(topics, ensure_ascii=False), question_pattern),
        )


def fetch_logs(limit: int = 50) -> list[dict]:
    """최근 대화 로그를 최신순으로 반환 (사이드바 통계용)."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversation_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_topic_stats() -> list[dict]:
    """주제별 질문 빈도 집계."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT topics, COUNT(*) as cnt FROM conversation_logs GROUP BY topics ORDER BY cnt DESC"
        ).fetchall()
    # topics 는 JSON 배열이므로 파싱해서 개별 주제로 집계
    counts: dict[str, int] = {}
    for row in rows:
        for topic in json.loads(row["topics"]):
            counts[topic] = counts.get(topic, 0) + row["cnt"]
    return sorted([{"topic": t, "count": c} for t, c in counts.items()], key=lambda x: -x["count"])
