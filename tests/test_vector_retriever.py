"""
test_vector_retriever.py

Testea la lógica de construcción de filtros (_build_filter), que es pura
y no requiere Ollama ni un índice Chroma real -- MarvelRetriever._build_filter
es un @staticmethod, así que se puede probar sin instanciar la clase (evita
la dependencia de Ollama/Chroma en el conjunto de tests rápidos).

Los tests de search()/search_with_scores() en sí (que sí requieren un
índice real) se dejan fuera de este archivo deliberadamente -- se cubren
con las pruebas manuales end-to-end documentadas en el README (Fase 1),
no con pytest, porque requieren Ollama corriendo y un modelo de embeddings
descargado, algo que no se puede asumir en un entorno de CI genérico.
"""

from src.retrieval.vector_retriever import MarvelRetriever


class TestBuildFilter:
    def test_no_filters_returns_none(self):
        assert MarvelRetriever._build_filter({}) is None

    def test_single_filter_returns_simple_dict(self):
        result = MarvelRetriever._build_filter({"mcu_phase": "Phase 4"})
        assert result == {"mcu_phase": "Phase 4"}

    def test_multiple_filters_use_and_operator(self):
        result = MarvelRetriever._build_filter({"mcu_phase": "Phase 4", "type": "series"})
        assert "$and" in result
        assert {"mcu_phase": "Phase 4"} in result["$and"]
        assert {"type": "series"} in result["$and"]