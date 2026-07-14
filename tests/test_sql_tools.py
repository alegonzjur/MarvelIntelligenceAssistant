"""
test_sql_tools.py

Formaliza las pruebas manuales que se fueron haciendo por consola durante
el desarrollo de sql_tools.py -- incluye el caso de regresión real
detectado en producción (mcu_phase="None" como string literal, ver
hyb-02 en el reporte de evaluación de la Fase 4).
"""

import pytest

from tests.conftest import requires_db
from src.retrieval import sql_tools


@requires_db
class TestTopByRevenue:
    def test_returns_results_for_known_phase(self):
        result = sql_tools.top_by_revenue(mcu_phase="Phase 4", n=1)
        assert "Doctor Strange" in result or "Spider-Man" in result or "Black Panther" in result

    def test_no_results_for_nonexistent_phase(self):
        # "Phase 5" solo (sin "/6") no existe en el dataset -- ver known_data_caveats
        # en golden_questions.json.
        result = sql_tools.top_by_revenue(mcu_phase="Phase 5", n=1)
        assert "No se encontraron resultados" in result


@requires_db
class TestTopByBudget:
    def test_highest_budget_phase4(self):
        result = sql_tools.top_by_budget(mcu_phase="Phase 4", n=1)
        assert "Doctor Strange in the Multiverse of Madness" in result
        assert "$290,000,000" in result


@requires_db
class TestTopByRoi:
    def test_best_roi_overall(self):
        result = sql_tools.top_by_roi(n=1)
        assert "Deadpool" in result
        assert "2016" in result

    def test_regression_string_none_is_treated_as_no_filter(self):
        """Regresión real (Fase 4, hyb-02): el LLM a veces pasa mcu_phase
        como el STRING literal "None" en vez de un null real. Sin
        normalizar, esto generaba `WHERE mcu_phase = 'None'` -- cero
        resultados, fallo silencioso. Debe comportarse como si no se
        hubiera pasado ningún filtro."""
        with_string_none = sql_tools.top_by_roi(mcu_phase="None", n=1)
        without_filter = sql_tools.top_by_roi(mcu_phase=None, n=1)
        assert with_string_none == without_filter
        assert "No se encontraron resultados" not in with_string_none


@requires_db
class TestActorAppearances:
    def test_returns_ranked_list(self):
        result = sql_tools.actor_appearances(n=5)
        lines = result.strip().split("\n")
        assert len(lines) == 5
        assert lines[0].startswith("1.")


@requires_db
class TestTitlesByActor:
    def test_partial_case_insensitive_match(self):
        result = sql_tools.titles_by_actor("robert downey")
        assert "Iron Man" in result

    def test_unknown_actor_returns_message(self):
        result = sql_tools.titles_by_actor("Nombre Que No Existe De Verdad")
        assert "No se encontró" in result


@requires_db
class TestBestRated:
    def test_case_insensitive_source(self):
        upper = sql_tools.best_rated(source="IMDb", n=1)
        lower = sql_tools.best_rated(source="imdb", n=1)
        assert upper == lower

    def test_top_imdb_is_xmen97(self):
        result = sql_tools.best_rated(source="IMDb", n=1)
        assert "X-Men" in result


@requires_db
class TestCountTitles:
    def test_phase4_series_count(self):
        assert sql_tools.count_titles(mcu_phase="Phase 4", media_type="series") == "1"

    def test_non_mcu_movie_count(self):
        assert sql_tools.count_titles(mcu_phase="Non-MCU", media_type="movie") == "21"

    def test_string_none_normalized(self):
        """Mismo caso de regresión que en top_by_roi, pero para count_titles,
        que tiene DOS parámetros opcionales de tipo string."""
        assert sql_tools.count_titles(mcu_phase="None", media_type="null") == sql_tools.count_titles()