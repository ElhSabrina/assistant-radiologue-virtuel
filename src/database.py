from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
DEFAULT_DB   = Path(__file__).resolve().parents[1] / "eval" / "predictions.sqlite"


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


def insert_run(
    image_name: str,
    prediction: dict,
    db_path: str | Path = DEFAULT_DB,
) -> int:
    """Insert one prediction into `runs` and return its row id."""
    init_db(db_path)
    conn = connect(db_path)
    cur = conn.execute(
        """
        INSERT INTO runs(
            case_id, image_path, model_name, prompt_version,
            prediction_json, predicted_class, confidence, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_name,
            image_name,
            prediction.get("model_name"),
            prediction.get("prompt_version"),
            json.dumps(prediction, ensure_ascii=False),
            prediction.get("predicted_class"),
            float(prediction.get("confidence", 0.0)),
            int(prediction.get("latency_ms", 0)),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_recent_runs(
    limit: int = 100,
    db_path: str | Path = DEFAULT_DB,
) -> list[dict]:
    """Return the most recent `limit` runs as plain dicts."""
    if not Path(db_path).exists():
        return []
    conn = connect(db_path)
    rows = conn.execute(
        """
        SELECT id, case_id, model_name, prompt_version,
               predicted_class, confidence, latency_ms, created_at
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(db_path: str | Path = DEFAULT_DB) -> dict:
    """Aggregate stats over all stored runs."""
    if not Path(db_path).exists():
        return {}
    conn = connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    dist  = conn.execute(
        "SELECT predicted_class, COUNT(*) AS n FROM runs GROUP BY predicted_class"
    ).fetchall()
    avg_conf = conn.execute("SELECT AVG(confidence) FROM runs").fetchone()[0]
    avg_lat  = conn.execute("SELECT AVG(latency_ms) FROM runs").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "distribution": {r["predicted_class"]: r["n"] for r in dist},
        "avg_confidence": round(avg_conf or 0, 3),
        "avg_latency_ms": round(avg_lat or 0, 1),
    }
