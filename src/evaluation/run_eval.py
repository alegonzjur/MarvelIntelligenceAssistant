"""
run_eval.py

Evalúa el pipeline completo (router.py + chain.py) contra las 19 golden
questions de golden_questions.json.

Decisión de diseño: NO se usa RAGAS. RAGAS requiere un LLM juez (normalmente
vía API externa) para calcular métricas como faithfulness o context
precision, lo cual introduce una dependencia externa que rompe el enfoque
100% local del proyecto. En su lugar, se implementan tres chequeos propios,
deterministas y baratos de calcular:

1. Routing accuracy: ¿la categoría predicha por el router coincide con la
   categoría real de la pregunta (narrative/analytical/hybrid)?
2. Retrieval hit-rate: para preguntas narrativas e híbridas, ¿el documento
   fuente esperado (expected_source) aparece entre los documentos
   recuperados?
3. Tool match: para preguntas analíticas e híbridas, ¿la tool que
   finalmente se ejecutó coincide con expected_tool?

Esto NO mide la calidad de la REDACCIÓN final (fluidez, tono, si "suena
bien") -- para eso, la respuesta generada se guarda en el reporte para
revisión manual. Mide si el pipeline recuperó/usó la fuente correcta, que
es lo que importa para detectar alucinaciones o fallos de routing.

Uso:
    python -m src.evaluation.run_eval
"""

# Importación de librerías
import json
from pathlib import Path
from typing import Any
from src.orchestration.chain import answer

# Rutas
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


# Carga de preguntas golden
def load_golden_questions() -> list[dict[str, Any]]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["questions"]


# Funciones auxiliares
def _source_in_documents(expected_source: dict, documents: list) -> bool:
    return any(
        doc.metadata.get("title") == expected_source["title"]
        and doc.metadata.get("year") == expected_source["year"]
        for doc in documents
    )

# Extrae el nombre de la primera tool efectivamente invocada
def _tool_used(raw_context: dict) -> str | None:
    """Extrae el nombre de la primera tool efectivamente invocada, tanto en
    respuestas analíticas puras como en el paso analítico de una híbrida."""
    results = raw_context.get("tool_results") or raw_context.get("analytical_step") or []
    if not results:
        return None
    return results[0].get("tool")


# Evaluación de preguntas narrativas
def evaluate_narrative(item: dict, result: dict) -> dict[str, Any]:
    raw = result["raw_context"]
    hit = _source_in_documents(item["expected_source"], raw.get("documents", []))
    return {"retrieval_hit": hit}


# Evaluación de preguntas analíticas
def evaluate_analytical(item: dict, result: dict) -> dict[str, Any]:
    raw = result["raw_context"]
    tool_used = _tool_used(raw)
    expected_tool = item.get("expected_tool", "")
    # expected_tool puede venir como "run_safe_sql_query (fallback)" -- se
    # compara por contención, no igualdad exacta.
    tool_match = bool(tool_used) and tool_used in expected_tool
    return {"tool_used": tool_used, "tool_match": tool_match}


# Evaluación de preguntas híbridas
def evaluate_hybrid(item: dict, result: dict) -> dict[str, Any]:
    raw = result["raw_context"]
    steps = item["steps"]
    narrative_step = next(s for s in steps if "expected_source" in s)

    tool_used = _tool_used(raw)
    retrieval_hit = _source_in_documents(narrative_step["expected_source"], raw.get("documents", []))

    return {
        "tool_used": tool_used,
        "retrieval_hit": retrieval_hit,
        "identified_target": raw.get("identified_target"),
        "analytical_step": raw.get("analytical_step"),
    }


# Ejecución de la evaluación
def run_eval() -> list[dict[str, Any]]:
    questions = load_golden_questions()
    report = []
    # Iteramos sobre cada pregunta golden
    for item in questions:
        print(f"Evaluando {item['id']}...", end=" ", flush=True)
        result = answer(item["question"])
        predicted_category = str(result["category"].value)
        category_match = predicted_category == item["category"]
        # Construimos el diccionario de resultados
        entry: dict[str, Any] = {
            "id": item["id"],
            "category": item["category"],
            "predicted_category": predicted_category,
            "category_match": category_match,
            "question": item["question"],
            "answer": result["answer"],
        }

        # Evaluamos según la categoría
        if item["category"] == "narrative":
            entry.update(evaluate_narrative(item, result))
        elif item["category"] == "analytical":
            entry.update(evaluate_analytical(item, result))
        else:
            entry.update(evaluate_hybrid(item, result))

        report.append(entry)
        print("OK" if category_match else "MISROUTED")

    return report


# Resumen de la evaluación
def summarize(report: list[dict[str, Any]]) -> None:
    total = len(report)
    routing_ok = sum(1 for r in report if r["category_match"])

    print(f"\n{'=' * 60}")
    print(f"Routing accuracy: {routing_ok}/{total} ({routing_ok/total:.0%})")
    # Evaluamos cada categoría por separado
    narrative = [r for r in report if r["category"] == "narrative"]
    if narrative:
        hits = sum(1 for r in narrative if r.get("retrieval_hit"))
        print(f"Retrieval hit-rate (narrative): {hits}/{len(narrative)} ({hits/len(narrative):.0%})")

    analytical = [r for r in report if r["category"] == "analytical"]
    if analytical:
        matches = sum(1 for r in analytical if r.get("tool_match"))
        print(f"Tool match (analytical): {matches}/{len(analytical)} ({matches/len(analytical):.0%})")

    hybrid = [r for r in report if r["category"] == "hybrid"]
    if hybrid:
        hits = sum(1 for r in hybrid if r.get("retrieval_hit"))
        print(f"Retrieval hit-rate (hybrid): {hits}/{len(hybrid)} ({hits/len(hybrid):.0%})")

    misrouted = [r["id"] for r in report if not r["category_match"]]
    if misrouted:
        print(f"\nPreguntas mal enrutadas: {misrouted}")

    failed_retrieval = [
        r["id"] for r in report if r.get("retrieval_hit") is False
    ]
    if failed_retrieval:
        print(f"Preguntas con fallo de retrieval: {failed_retrieval}")

    failed_tool = [r["id"] for r in report if r.get("tool_match") is False]
    if failed_tool:
        print(f"Preguntas con tool incorrecta: {failed_tool}")

# Punto de entrada
if __name__ == "__main__":
    report = run_eval()
    summarize(report)

    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nReporte completo guardado en {RESULTS_PATH}")