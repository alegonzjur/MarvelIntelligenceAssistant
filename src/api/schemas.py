"""
schemas.py

Modelos Pydantic de request/response de la API. Separados de main.py para
poder reutilizarlos (tests, documentación OpenAPI, futuro cliente) sin
importar FastAPI entero.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pregunta en lenguaje natural sobre el catálogo MCU/Marvel.",
        examples=["¿De qué trata Loki?", "¿Cuál es la película con más recaudación?"],
    )


class SourceItem(BaseModel):
    title: str
    year: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    category: str = Field(description="'narrative', 'analytical' o 'hybrid'")
    routing_reasoning: str | None = Field(
        default=None, description="Justificación del router para esta clasificación."
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Documentos narrativos recuperados que sustentan la respuesta "
        "(vacío si la pregunta era puramente analítica).",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Nombres de las tools SQL invocadas para resolver la parte analítica "
        "(vacío si la pregunta era puramente narrativa).",
    )


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    database_ready: bool