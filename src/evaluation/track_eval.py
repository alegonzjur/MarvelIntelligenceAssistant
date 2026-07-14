"""
track_eval.py

Envuelve run_eval.py con tracking en MLflow (backend local basado en
ficheros, sin servidor), para poder comparar métricas de evaluación entre
iteraciones -- cambios de prompt, de modelo (ej. llama3.2:3b vs llama3.1),
de tools -- sin perder el historial de cada intento.

Decisión de diseño: se reutiliza el MISMO evaluador ligero de run_eval.py
(la razón de no usar RAGAS está documentada allí). Este script no
reimplementa lógica de evaluación, solo envuelve sus métricas ya
calculadas en un mlflow.start_run() y adjunta el reporte completo como
artifact para poder inspeccionar respuesta por respuesta más tarde.

Uso:
    python -m src.evaluation.track_eval
    python -m src.evaluation.track_eval --run-name "llama3.1-baseline"

Ver resultados (backend SQLite local, no requiere servidor MLflow):
    mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

from src.evaluation.run_eval import RESULTS_PATH, run_eval
from src.orchestration.chain import SYNTHESIS_MODEL
from src.orchestration.router import ROUTER_MODEL

MLFLOW_DIR = Path(__file__).resolve().parents[2] / "mlflow"
MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"
EXPERIMENT_NAME = "marvel-rag-hybrid-eval"


def _git_commit() -> str | None:
    """Best-effort: si no hay git disponible o no es un repo, se omite
    el parámetro en vez de fallar el tracking entero por esto."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def compute_metrics(report: list[dict[str, Any]]) -> dict[str, float]:
    """Misma lógica que summarize() en run_eval.py, pero devolviendo un
    dict de métricas en vez de imprimir -- para poder loguearlas en MLflow."""
    total = len(report)
    routing_ok = sum(1 for r in report if r["category_match"])
    metrics: dict[str, float] = {
        "total_questions": total,
        "routing_accuracy": routing_ok / total,
    }

    narrative = [r for r in report if r["category"] == "narrative"]
    if narrative:
        hits = sum(1 for r in narrative if r.get("retrieval_hit"))
        metrics["retrieval_hit_rate_narrative"] = hits / len(narrative)

    analytical = [r for r in report if r["category"] == "analytical"]
    if analytical:
        matches = sum(1 for r in analytical if r.get("tool_match"))
        metrics["tool_match_analytical"] = matches / len(analytical)

    hybrid = [r for r in report if r["category"] == "hybrid"]
    if hybrid:
        hits = sum(1 for r in hybrid if r.get("retrieval_hit"))
        metrics["retrieval_hit_rate_hybrid"] = hits / len(hybrid)

    return metrics


def track(run_name: str | None = None) -> dict[str, float]:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = run_name or f"{SYNTHESIS_MODEL}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

    print("Ejecutando la evaluación completa (puede tardar varios minutos)...")
    report = run_eval()
    metrics = compute_metrics(report)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("router_model", ROUTER_MODEL)
        mlflow.log_param("synthesis_model", SYNTHESIS_MODEL)
        git_commit = _git_commit()
        if git_commit:
            mlflow.log_param("git_commit", git_commit)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        RESULTS_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        mlflow.log_artifact(str(RESULTS_PATH))

    print(f"\nRun '{run_name}' registrado en MLflow.")
    print(f"Ver resultados: mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB_PATH}")
    print(f"\nMétricas:\n{json.dumps(metrics, indent=2, ensure_ascii=False)}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre del run en MLflow (por defecto: modelo + timestamp UTC)",
    )
    args = parser.parse_args()
    track(run_name=args.run_name)