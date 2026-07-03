from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.guardrails import validate_prediction, detect_hallucination
from src.metrics import summarize_metrics
from src.database import insert_run, init_db
from src.pipeline import predict as predict_one


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(mode: str, db_path: Path, engine: str, cases_csv: Path, limit: int | None) -> tuple[list[dict], dict]:
    cases = read_cases(cases_csv)
    if limit is not None:
        cases = cases[:limit]
    rows = []
    init_db(db_path)
    for case in cases:
        image_path = ROOT / case['image_path']
        pred = predict_one(image_path, mode=mode, engine=engine)
        valid, errors = validate_prediction(pred)
        hallucinated = detect_hallucination(pred)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': valid,
            'hallucination': hallucinated,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'justification': pred.get('justification', ''),
            'guardrail_errors': ';'.join(errors),
        }
        rows.append(row)
        insert_run(db_path, case['case_id'], str(image_path), pred)
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs improved evaluation.")
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved', 'screening'], default='toy',
                        help="toy runs baseline and improved back to back")
    parser.add_argument('--engine', choices=['toy', 'medgemma'], default='toy',
                        help="toy = deterministic rule engine; medgemma = real VLM")
    parser.add_argument('--cases', type=Path, default=ROOT / 'data' / 'synthetic_cases.csv',
                        help="CSV of cases (case_id,image_path,label,...)")
    parser.add_argument('--limit', type=int, default=None, help="cap number of cases (debug)")
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'results')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'eval' / 'results' / 'runs.sqlite')
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ['baseline', 'improved'] if args.mode == 'toy' else [args.mode]
    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.db_path, args.engine, args.cases, args.limit)
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        (out_dir / f'{mode}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        summary.append({'mode': mode, 'engine': args.engine, **metrics})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
