"""
main.py

API FastAPI que expone el pipeline completo (router + chain) como un
endpoint HTTP. Sin estado propio: cada request llama a chain.answer(),
que a su vez orquesta clasificación, recuperación y síntesis.

Nota de diseño: los endpoints se declaran con `def` normal, no `async def`.
chain.answer() hace llamadas bloqueantes a Ollama (vía requests HTTP
síncronos), así que si se declarara `async def` sin usar un cliente async,
bloquearía el event loop entero y las requests concurrentes se encolarían
en vez de procesarse en paralelo. Con `def` normal, FastAPI ejecuta el
handler en un threadpool automáticamente, lo cual es el comportamiento
correcto aquí.

Uso:
    uvicorn src.api.main:app --reload --port 8000

Documentación interactiva (Swagger UI) una vez arrancado:
    http://localhost:8000/docs
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data_prep.build_sqlite import DB_PATH
from src.orchestration.chain import answer as chain_answer
from src.orchestration.router import ROUTER_MODEL
from src.orchestration.chain import SYNTHESIS_MODEL
from src.retrieval.vector_retriever import PERSIST_DIR

from src.api.schemas import AskRequest, AskResponse, HealthResponse, SourceItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marvel_api")

app = FastAPI(
    title="Marvel Intelligence Assistant API",
    description=(
        "API híbrida RAG + SQL agent sobre el catálogo MCU/Marvel. "
        "Clasifica cada pregunta como narrativa, analítica o híbrida, y "
        "recupera el contexto de la fuente correcta antes de sintetizar "
        "la respuesta final."
    ),
    version="0.1.0",
)

# CORS abierto para desarrollo local (el frontend Streamlit corre en otro
# puerto). Restringir a un origin concreto antes de exponer esto fuera de
# localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_sources(raw_context: dict[str, Any]) -> list[SourceItem]:
    documents = raw_context.get("documents") or []
    return [
        SourceItem(title=doc.metadata.get("title", "?"), year=doc.metadata.get("year"))
        for doc in documents
    ]


def _extract_tools_used(raw_context: dict[str, Any]) -> list[str]:
    results = raw_context.get("tool_results") or raw_context.get("analytical_step") or []
    return [r["tool"] for r in results if r.get("tool")]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Chequeo básico: confirma que el índice vectorial y la base de datos
    existen en disco. NO verifica que Ollama esté accesible (eso solo se
    sabe al intentar responder una pregunta de verdad)."""
    return HealthResponse(
        status="ok",
        vectorstore_ready=PERSIST_DIR.exists(),
        database_ready=DB_PATH.exists(),
    )


@app.get("/info")
def info() -> dict[str, str]:
    """Metadatos de configuración, útiles para depurar qué modelos está
    usando la API sin tener que mirar el código."""
    return {"router_model": ROUTER_MODEL, "synthesis_model": SYNTHESIS_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = chain_answer(request.question)
    except Exception as exc:  # noqa: BLE001 -- se traduce a 500 genérico;
        # el detalle completo va a logs, no a la respuesta HTTP, para no
        # filtrar trazas internas (rutas de archivo, nombres de tools) al
        # cliente.
        logger.exception("Error resolviendo la pregunta: %r", request.question)
        raise HTTPException(
            status_code=500,
            detail="Error interno al procesar la pregunta. Revisa los logs del servidor.",
        ) from exc

    raw_context = result["raw_context"]
    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        category=result["category"].value,
        routing_reasoning=result.get("routing_reasoning"),
        sources=_extract_sources(raw_context),
        tools_used=_extract_tools_used(raw_context),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)