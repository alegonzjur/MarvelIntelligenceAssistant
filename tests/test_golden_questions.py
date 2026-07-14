"""
test_golden_questions.py

Valida la ESTRUCTURA de golden_questions.json (no las respuestas en sí,
eso lo hace run_eval.py con el LLM real). Sirve de red de seguridad para
que un error de edición manual del JSON (como el que se corrigió al
recategorizar hyb-03 -> anl-10) se detecte al vuelo con pytest, en vez de
descubrirse a mitad de una ejecución larga de run_eval.py.
"""

import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "src" / "evaluation" / "golden_questions.json"

VALID_CATEGORIES = {"narrative", "analytical", "hybrid"}


def _load():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


class TestGoldenQuestionsStructure:
    def test_file_is_valid_json(self):
        _load()  # no debe lanzar

    def test_has_at_least_one_question_per_category(self):
        data = _load()
        categories = {q["category"] for q in data["questions"]}
        assert categories == VALID_CATEGORIES

    def test_ids_are_unique(self):
        data = _load()
        ids = [q["id"] for q in data["questions"]]
        assert len(ids) == len(set(ids))

    def test_narrative_questions_have_expected_source(self):
        data = _load()
        for q in data["questions"]:
            if q["category"] == "narrative":
                assert "expected_source" in q, f"{q['id']} sin expected_source"
                assert "title" in q["expected_source"]
                assert "year" in q["expected_source"]

    def test_hybrid_questions_have_steps_with_narrative_target(self):
        data = _load()
        for q in data["questions"]:
            if q["category"] == "hybrid":
                assert "steps" in q, f"{q['id']} sin steps"
                narrative_steps = [s for s in q["steps"] if "expected_source" in s]
                assert len(narrative_steps) >= 1, f"{q['id']} no tiene paso narrativo con expected_source"

    def test_all_categories_are_valid(self):
        data = _load()
        for q in data["questions"]:
            assert q["category"] in VALID_CATEGORIES, f"{q['id']} tiene categoría inválida: {q['category']}"