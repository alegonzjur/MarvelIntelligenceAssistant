"""
app.py

Frontend Streamlit del Marvel Intelligence Assistant. Interfaz tipo chat
que consume la API FastAPI (src/api/main.py) vía HTTP -- deliberadamente
NO importa src.orchestration.chain directamente, para mantener frontend y
backend desacoplados (el frontend podría correr en otra máquina, y la API
se puede probar/documentar de forma independiente vía Swagger).

Requiere la API corriendo en local:
    uvicorn src.api.main:app --reload --port 8000

Uso:
    streamlit run frontend/app.py
"""

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("MARVEL_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("MARVEL_REQUEST_TIMEOUT", "300"))  # 5 min: una
    # pregunta híbrida encadena varias llamadas al LLM (clasificación, tool
    # calling, extracción, síntesis), y en hardware modesto con el modelo
    # recién cargado en memoria (cold start) puede superar de sobra los 120s
    # que se usaban antes. Configurable por si tu equipo lo necesita aún
    # más alto: MARVEL_REQUEST_TIMEOUT=600 streamlit run frontend/app.py

GOLDEN_QUESTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "evaluation" / "golden_questions.json"
)

CATEGORY_LABELS = {
    "narrative": "📖 Narrativa",
    "analytical": "📊 Analítica",
    "hybrid": "🔀 Híbrida",
}

st.set_page_config(
    page_title="Marvel Intelligence Assistant",
    page_icon="🦸",
    layout="centered",
)


def ask_api(question: str) -> dict:
    response = requests.post(
        f"{API_URL}/ask", json={"question": question}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def check_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data
def load_example_questions(n_per_category: int = 2) -> list[str]:
    """Un puñado de preguntas de ejemplo sacadas de golden_questions.json,
    una o dos por categoría, para que la demo no empiece con una pantalla
    en blanco. Si el fichero no existe (ej. despliegue sin el repo
    completo), se devuelve una lista vacía sin romper la app."""
    if not GOLDEN_QUESTIONS_PATH.exists():
        return []
    data = json.loads(GOLDEN_QUESTIONS_PATH.read_text(encoding="utf-8"))
    by_category: dict[str, list[str]] = {}
    for item in data.get("questions", []):
        by_category.setdefault(item["category"], []).append(item["question"])
    examples = []
    for category, questions in by_category.items():
        examples.extend(questions[:n_per_category])
    return examples


def render_assistant_message(content: dict) -> None:
    category = content.get("category")
    if category in CATEGORY_LABELS:
        st.caption(CATEGORY_LABELS[category])

    st.markdown(content["answer"])

    sources = content.get("sources") or []
    tools_used = content.get("tools_used") or []

    if sources or tools_used:
        with st.expander("Ver de dónde viene esta respuesta"):
            if sources:
                st.markdown("**Documentos recuperados:**")
                for s in sources:
                    st.markdown(f"- {s['title']} ({s.get('year', '?')})")
            if tools_used:
                st.markdown("**Herramientas analíticas usadas:**")
                for t in tools_used:
                    st.markdown(f"- `{t}`")
            if content.get("routing_reasoning"):
                st.caption(f"Razonamiento del router: {content['routing_reasoning']}")


def handle_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Consultando el catálogo..."):
        try:
            result = ask_api(question)
            st.session_state.messages.append({"role": "assistant", "content": result})
        except requests.exceptions.Timeout:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "answer": (
                            f"La API tardó más de {REQUEST_TIMEOUT}s en responder. Si es la "
                            "primera pregunta después de un rato sin usar el proyecto, "
                            "puede que Ollama esté cargando el modelo en memoria por "
                            "primera vez (cold start) -- prueba a repetir la pregunta, "
                            "normalmente la segunda vez es mucho más rápida. Si sigue "
                            "fallando, comprueba que Ollama está corriendo "
                            "(`curl http://localhost:11434`)."
                        ),
                        "category": None,
                        "sources": [],
                        "tools_used": [],
                    },
                }
            )
        except requests.exceptions.ConnectionError:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "answer": (
                            f"No puedo conectar con la API en `{API_URL}`. "
                            "¿Está corriendo `uvicorn src.api.main:app --reload --port 8000`?"
                        ),
                        "category": None,
                        "sources": [],
                        "tools_used": [],
                    },
                }
            )
        except requests.exceptions.HTTPError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "answer": f"La API devolvió un error ({exc.response.status_code}). "
                        "Revisa los logs del servidor de la API.",
                        "category": None,
                        "sources": [],
                        "tools_used": [],
                    },
                }
            )


# --- Sidebar ---
with st.sidebar:
    st.title("🦸 Marvel Intelligence Assistant")
    st.caption("RAG + SQL agent híbrido sobre el catálogo MCU/Marvel")

    health = check_health()
    if health is None:
        st.error(f"API no disponible en {API_URL}")
    else:
        st.success("API conectada")
        if not health.get("vectorstore_ready"):
            st.warning("Índice vectorial no encontrado. Ejecuta embed_index.py.")
        if not health.get("database_ready"):
            st.warning("Base de datos no encontrada. Ejecuta build_sqlite.py.")

    st.divider()
    st.subheader("Preguntas de ejemplo")
    for q in load_example_questions():
        if st.button(q, use_container_width=True, key=f"example_{q}"):
            handle_question(q)

    st.divider()
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Historial de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message["content"])
        else:
            st.markdown(message["content"])

# --- Input de chat ---
if prompt := st.chat_input("Pregunta algo sobre el catálogo MCU/Marvel..."):
    handle_question(prompt)
    st.rerun()