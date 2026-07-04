from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile

from src.pipeline import predict
from src.database import insert_run

app = FastAPI(title="Assistant radiologue virtuel EFREI", version="0.2.0")
UPLOAD_DIR = Path("tmp_uploads")
# Interactive logging is opt-in: set ASSISTANT_DB to a path to persist API runs.
# (The batch evaluation always logs to its own --db-path.) Kept off by default so
# a running server never leaves a database at the repository root.
DB_PATH = os.environ.get("ASSISTANT_DB")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis"}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("improved"),
    engine: str = Form("toy"),
) -> dict:
    """Analyse an uploaded chest X-ray.

    ``engine`` selects the toy rule engine (default, instant) or ``medgemma``
    (real VLM, slower, needs the model available). ``mode`` is baseline/improved.
    Every call is logged to SQLite.
    """
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    pred = predict(target, mode=mode, engine=engine)
    if DB_PATH:
        try:
            insert_run(DB_PATH, f"upload_{safe_stem}", str(target), pred)
        except Exception:
            # Logging must never break a prediction response.
            pass
    return pred
