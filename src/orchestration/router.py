"""
router.py

Punto de entrada de la orquestación: clasifica cada pregunta entrante y
recupera el CONTEXTO necesario (documentos del retriever, resultados de
tools SQL, o ambos encadenados). No genera la respuesta final en lenguaje
natural -- eso es responsabilidad de chain.py, que toma el contexto
devuelto aquí y lo pasa a un prompt de síntesis.

Esta separación es deliberada: permite evaluar la calidad del ROUTING
(¿recuperó el contexto correcto?) independientemente de la calidad de la
REDACCIÓN final del LLM -- son dos fuentes de error distintas y conviene
poder medirlas por separado en la Fase de evaluación.

Requiere un modelo de Ollama con soporte de tool calling (ej. llama3.1).
No probado en este entorno por no disponer de Ollama; probar en local con:

    python -m src.orchestration.router
"""
# Importación de librerías
from enum import Enum
from typing import Any
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from src.retrieval.sql_agent import get_all_tools
from src.retrieval.vector_retriever import MarvelRetriever

# Configuración del modelo
ROUTER_MODEL = "llama3.2:3b"  # debe soportar tool calling / structured output

# Definición de categorías
class RouteCategory(str, Enum):
    NARRATIVE = "narrative"
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"


# Definición del modelo de decisión
class RouteDecision(BaseModel):
    category: RouteCategory = Field(
        description="Tipo de pregunta: 'narrative' (trama, personajes, argumento), "
        "'analytical' (cifras, rankings, conteos, comparaciones), o 'hybrid' "
        "(requiere primero identificar un título mediante datos analíticos y "
        "después preguntar algo narrativo sobre ese título concreto)."
    )
    reasoning: str = Field(description="Justificación breve de la clasificación, en una frase.")


CLASSIFIER_SYSTEM_PROMPT = """Eres un router que clasifica preguntas sobre un catálogo \
de películas y series de Marvel/MCU en tres categorías:

- narrative: preguntas sobre trama, personajes, argumento, tono, conflicto de una \
  obra concreta. Ej: "¿de qué trata X?", "¿qué le pasa a Y en la serie Z?".
- analytical: preguntas sobre cifras, rankings, conteos, comparaciones agregadas. \
  Ej: "¿cuál es la película que más ha recaudado?", "¿cuántas series tiene Phase 4?".
- hybrid: preguntas que combinan ambas: primero hay que IDENTIFICAR un título \
  mediante un criterio analítico (el más caro, el mejor valorado, el de más ROI...) \
  y LUEGO preguntar algo narrativo sobre ese título concreto. \
  Ej: "de las películas de Phase 4 con más presupuesto, ¿de qué trata la más cara?".

Señal clave para distinguir 'analytical' de 'hybrid': si la pregunta pide una CIFRA \
o LISTA como respuesta final, es analytical. Si pide TRAMA/CONTENIDO de un título que \
hay que identificar primero por un criterio numérico, es hybrid.
"""

# Función para obtener el LLM
def _get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=ROUTER_MODEL, temperature=temperature)


# Función para clasificar la pregunta
def classify(question: str, llm: ChatOllama | None = None) -> RouteDecision:
    llm = llm or _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    return structured_llm.invoke(
        [
            HumanMessage(content=f"{CLASSIFIER_SYSTEM_PROMPT}\n\nPregunta: {question}")
        ]
    )


# Función para ejecutar el paso de llamada a herramientas
def _run_tool_calling_step(question: str, llm: ChatOllama) -> list[dict[str, Any]]:
    """Deja que el LLM elija y ejecute una o varias tools SQL para responder
    la parte analítica de la pregunta. Devuelve los resultados crudos de
    cada tool invocada (no la redacción del LLM)."""
    tools = get_all_tools()
    tools_by_name = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(
        f"Responde a esta pregunta analítica usando las tools disponibles. "
        f"Pregunta: {question}"
    )

    results = []
    for call in response.tool_calls:
        tool = tools_by_name.get(call["name"])
        if tool is None:
            results.append({"tool": call["name"], "error": "tool desconocida"})
            continue
        output = tool.invoke(call["args"])
        results.append({"tool": call["name"], "args": call["args"], "output": output})

    if not results:
        # El LLM no llamó a ninguna tool -- se propaga tal cual para que
        # chain.py pueda decidir cómo manejar el caso (reintento, mensaje
        # de "no se pudo responder", etc.)
        results.append({"tool": None, "output": response.content})

    return results


# Función para responder preguntas narrativas
def answer_narrative(question: str, retriever: MarvelRetriever, k: int = 4) -> dict[str, Any]:
    docs = retriever.search(question, k=k)
    return {
        "category": RouteCategory.NARRATIVE,
        "documents": docs,
        "sources": [{"title": d.metadata["title"], "year": d.metadata["year"]} for d in docs],
    }


# Función para responder preguntas analíticas
def answer_analytical(question: str, llm: ChatOllama) -> dict[str, Any]:
    tool_results = _run_tool_calling_step(question, llm)
    return {
        "category": RouteCategory.ANALYTICAL,
        "tool_results": tool_results,
    }


# Función para responder preguntas híbridas
def answer_hybrid(question: str, llm: ChatOllama, retriever: MarvelRetriever) -> dict[str, Any]:
    # Paso 1: resolver la parte analítica para identificar el/los título(s) objetivo.
    analytical_step = _run_tool_calling_step(question, llm)

    # Paso 2: extraer el título/año concreto del resultado de la tool, para
    # usarlo como query de recuperación narrativa. Se usa un LLM call corto
    # y estructurado en vez de regex, porque el formato de salida de las
    # tools varía (top-N con ranking, valor único, etc.).
    class ExtractedTarget(BaseModel):
        title: str = Field(description="Título exacto de la obra identificada, sin año.")
        year: int | None = Field(default=None, description="Año de la obra si se conoce.")

    tool_output_text = "\n".join(str(r.get("output", "")) for r in analytical_step)
    extractor = llm.with_structured_output(ExtractedTarget)
    target = extractor.invoke(
        f"A partir de este resultado analítico, extrae el título concreto sobre el que "
        f"habrá que buscar información narrativa después. Si hay varios, elige el primero "
        f"(el más relevante para la pregunta original).\n\n"
        f"Pregunta original: {question}\n\n"
        f"Resultado analítico:\n{tool_output_text}"
    )

    # Paso 3: recuperación narrativa filtrada por el título exacto identificado,
    # en vez de una búsqueda semántica libre -- ya sabemos qué documento
    # queremos, así que filtramos por metadata para evitar falsos positivos
    # de similitud con títulos parecidos.
    filters: dict[str, Any] = {"title": target.title}
    if target.year is not None:
        filters["year"] = target.year
    docs = retriever.search(question, k=3, **filters)

    if not docs:
        # Fallback: si el filtro exacto no encuentra nada (título mal
        # extraído o pequeña discrepancia de formato), se cae a búsqueda
        # semántica libre con el título como query.
        docs = retriever.search(target.title, k=3)

    return {
        "category": RouteCategory.HYBRID,
        "analytical_step": analytical_step,
        "identified_target": target.model_dump(),
        "documents": docs,
        "sources": [{"title": d.metadata["title"], "year": d.metadata["year"]} for d in docs],
    }


# Función para enrutar la pregunta
def route(question: str) -> dict[str, Any]:
    """Punto de entrada único: clasifica y recupera el contexto adecuado."""
    llm = _get_llm()
    decision = classify(question, llm=llm)
    retriever = MarvelRetriever()

    if decision.category == RouteCategory.NARRATIVE:
        result = answer_narrative(question, retriever)
    elif decision.category == RouteCategory.ANALYTICAL:
        result = answer_analytical(question, llm)
    else:
        result = answer_hybrid(question, llm, retriever)

    result["routing_reasoning"] = decision.reasoning
    return result



# Función para probar el router
if __name__ == "__main__":
    import json

    test_questions = [
        "¿De qué trata Spider-Man: Across the Spider-Verse?",
        "¿Cuál es la película con mayor recaudación del catálogo?",
        "De las películas de Phase 4, la que tuvo mayor presupuesto, ¿de qué trata?",
    ]

    for q in test_questions:
        print(f"\n=== {q} ===")
        result = route(q)
        print(f"Categoría: {result['category']}")
        print(f"Razonamiento: {result['routing_reasoning']}")
        if "sources" in result:
            print(f"Fuentes recuperadas: {result['sources']}")
        if "tool_results" in result:
            print(f"Tool results: {json.dumps(result['tool_results'], indent=2, default=str)}")
        if "analytical_step" in result:
            print(f"Paso analítico: {json.dumps(result['analytical_step'], indent=2, default=str)}")
        if "identified_target" in result:
            print(f"Título identificado para el paso narrativo: {result['identified_target']}")