"""
test_api.py

Tests de src/api/main.py usando FastAPI TestClient. chain.answer() se
mockea (vía unittest.mock.patch) porque llamarlo de verdad requiere Ollama
corriendo -- estos tests validan la CAPA HTTP (serialización, códigos de
estado, manejo de errores), no la calidad de las respuestas del LLM, que
se cubre en run_eval.py sobre las golden questions.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import src.api.main as main_module
from src.orchestration.router import RouteCategory


client = TestClient(main_module.app)


class TestHealthAndInfo:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "vectorstore_ready" in body
        assert "database_ready" in body

    def test_info_returns_model_names(self):
        response = client.get("/info")
        assert response.status_code == 200
        body = response.json()
        assert "router_model" in body
        assert "synthesis_model" in body


class TestAskEndpoint:
    def test_narrative_answer_serializes_correctly(self):
        fake_result = {
            "question": "¿De qué trata Loki?",
            "answer": "Loki sigue al dios del engaño...",
            "category": RouteCategory.NARRATIVE,
            "routing_reasoning": "Pregunta sobre trama.",
            "raw_context": {
                "documents": [
                    Document(page_content="...", metadata={"title": "Loki (Season 1)", "year": 2021})
                ],
            },
        }
        with patch.object(main_module, "chain_answer", return_value=fake_result):
            response = client.post("/ask", json={"question": "¿De qué trata Loki?"})

        assert response.status_code == 200
        body = response.json()
        assert body["category"] == "narrative"
        assert body["sources"] == [{"title": "Loki (Season 1)", "year": 2021}]
        assert body["tools_used"] == []

    def test_analytical_answer_serializes_tools_used(self):
        fake_result = {
            "question": "¿Cuál es la más taquillera?",
            "answer": "Avengers: Endgame.",
            "category": RouteCategory.ANALYTICAL,
            "routing_reasoning": "Pide una cifra.",
            "raw_context": {
                "tool_results": [{"tool": "top_by_revenue", "args": {}, "output": "..."}],
            },
        }
        with patch.object(main_module, "chain_answer", return_value=fake_result):
            response = client.post("/ask", json={"question": "¿Cuál es la más taquillera?"})

        assert response.status_code == 200
        body = response.json()
        assert body["tools_used"] == ["top_by_revenue"]
        assert body["sources"] == []

    def test_internal_error_returns_500_without_leaking_traceback(self):
        with patch.object(main_module, "chain_answer", side_effect=RuntimeError("fallo interno de Ollama")):
            response = client.post("/ask", json={"question": "¿De qué trata Loki?"})

        assert response.status_code == 500
        assert "fallo interno de Ollama" not in response.text

    def test_question_too_short_returns_422(self):
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 422

    def test_missing_question_field_returns_422(self):
        response = client.post("/ask", json={})
        assert response.status_code == 422