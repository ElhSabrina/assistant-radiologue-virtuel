from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import baseline_predict, improved_predict, toy_predict
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_run, init_db

Predictor = Callable[..., dict]

# (output label, predictor) pairs for each --mode.
# `toy` stays on the deterministic toy_predict so the CI smoke test is stable;
# `real` runs the actual étape-2 baseline_v1 vs improved_v1 comparison.
PLANS: dict[str, list[tuple[str, Predictor]]] = {
    'toy': [('baseline', toy_predict), ('improved', toy_predict)],
    'real': [('baseline', baseline_predict), ('improved', improved_predict)],
    'baseline': [('baseline', baseline_predict)],
    'improved': [('improved', improved_predict)],
}


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(label: str, predictor: Predictor, cases_path: Path, db_path: Path) -> tuple[list[dict], dict]:
    cases = read_cases(cases_path)
    rows = []
    init_db(db_path)
    for case in cases:
        image_path = ROOT / case['image_path']
        pred = apply_safety_guardrails(predictor(image_path, mode=label))
        valid, errors = validate_prediction(pred)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
        }
        rows.append(row)
        insert_run(image_name=case['case_id'], prediction=pred, db_path=db_path)
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list(PLANS), default='toy')
    parser.add_argument('--cases', type=Path, default=ROOT / 'data' / 'synthetic_cases.csv',
                        help="cases CSV to evaluate (e.g. data/rsna/cases.csv for the real RSNA set)")
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for label, predictor in PLANS[args.mode]:
        rows, metrics = run(label, predictor, args.cases, args.db_path)
        write_csv(out_dir / f'{label}_predictions.csv', rows)
        (out_dir / f'{label}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        summary.append({'mode': label, **metrics})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
