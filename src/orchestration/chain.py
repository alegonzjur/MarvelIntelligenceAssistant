"""
chain.py

Toma el contexto recuperado por router.route() (documentos narrativos,
resultados de tools SQL, o ambos) y genera la respuesta final en lenguaje
natural. Un prompt distinto por categoría, para no mezclar instrucciones
de "no inventes cifras" (analítica) con "no inventes trama" (narrativa).

Principio de diseño: el LLM de síntesis NUNCA decide qué fuente usar --
eso ya lo resolvió router.py. Aquí solo redacta a partir de lo que se le
entrega, lo cual reduce alucinaciones y hace más fácil depurar (si la
respuesta está mal, primero se comprueba si el CONTEXTO recuperado era
correcto -- ver router.py -- antes de sospechar de la redacción).

Uso:
    from src.orchestration.chain import answer
    result = answer("¿de qué trata Loki?")
    print(result["answer"])
"""

# Importación de librerías.
from typing import Any
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.config import OLLAMA_BASE_URL, SYNTHESIS_MODEL
from src.orchestration.router import RouteCategory, route

# Prompt para respuestas narrativas.
NARRATIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en el catálogo MCU/Marvel. Responde a la pregunta "
            "del usuario basándote ÚNICAMENTE en los documentos de contexto proporcionados. "
            "Si los documentos no contienen suficiente información para responder, dilo "
            "explícitamente en vez de inventar. Cita el título y año de la obra en tu "
            "respuesta. Responde en el mismo idioma en el que está formulada la pregunta.",
        ),
        (
            "human",
            "Pregunta: {question}\n\nDocumentos de contexto:\n{context}",
        ),
    ]
)

# Prompt para respuestas analíticas.
ANALYTICAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en el catálogo MCU/Marvel. Redacta una respuesta "
            "clara a partir del resultado de la consulta analítica proporcionada. "
            "NO inventes ni modifiques ninguna cifra, nombre o dato que no aparezca "
            "literalmente en el resultado. Si el resultado indica que no hay datos, "
            "dilo explícitamente. Responde en el mismo idioma en el que está formulada "
            "la pregunta.",
        ),
        (
            "human",
            "Pregunta: {question}\n\nResultado de la consulta:\n{context}",
        ),
    ]
)

# Prompt para respuestas híbridas.
HYBRID_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en el catálogo MCU/Marvel. La pregunta del usuario "
            "requiere combinar dos piezas de información: (1) un dato analítico que "
            "identifica un título concreto, y (2) contenido narrativo sobre ese título. "
            "El dato analítico YA HA SIDO CALCULADO y es la fuente de verdad: cópialo tal "
            "cual (cifra, título y año) para justificar por qué ese título es el relevante "
            "(ej. 'es el que más presupuesto tuvo, con $X'). NUNCA recalcules ROI, "
            "presupuesto, recaudación ni ninguna otra cifra a partir de los documentos "
            "narrativos, aunque estos mencionen números -- esos números son solo contexto "
            "de trama, no una fuente para recomputar el dato analítico. Si el título "
            "mencionado en los documentos narrativos no coincide con el identificado en "
            "el dato analítico, prevalece el dato analítico. Después, responde la parte "
            "narrativa basándote ÚNICAMENTE en los documentos de contexto, sin inventar "
            "trama. Responde en el mismo idioma en el que está formulada la pregunta.",
        ),
        (
            "human",
            "Pregunta: {question}\n\nDato analítico:\n{analytical_context}\n\n"
            "Documentos narrativos:\n{narrative_context}",
        ),
    ]
)


# Función para obtener el LLM.
def _get_llm(temperature: float = 0.3) -> ChatOllama:
    return ChatOllama(model=SYNTHESIS_MODEL, temperature=temperature, base_url=OLLAMA_BASE_URL)


# Función para formatear documentos.
def _format_documents(documents: list[Document]) -> str:
    if not documents:
        return "(sin documentos recuperados)"
    parts = []
    for doc in documents:
        title = doc.metadata.get("title", "?")
        year = doc.metadata.get("year", "?")
        parts.append(f"--- {title} ({year}) ---\n{doc.page_content}")
    return "\n\n".join(parts)


# Función para formatear resultados de herramientas.
def _format_tool_results(tool_results: list[dict[str, Any]]) -> str:
    if not tool_results:
        return "(sin resultados de herramientas)"
    parts = []
    for r in tool_results:
        tool_name = r.get("tool") or "respuesta_directa_del_modelo"
        parts.append(f"[{tool_name}]\n{r.get('output', '')}")
    return "\n\n".join(parts)


# Función para sintetizar respuestas narrativas.
def _synthesize_narrative(question: str, context: dict[str, Any], llm: ChatOllama) -> str:
    formatted = _format_documents(context["documents"])
    chain = NARRATIVE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": formatted})


# Función para sintetizar respuestas analíticas.
def _synthesize_analytical(question: str, context: dict[str, Any], llm: ChatOllama) -> str:
    formatted = _format_tool_results(context["tool_results"])
    chain = ANALYTICAL_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": formatted})


# Función para sintetizar respuestas híbridas.
def _synthesize_hybrid(question: str, context: dict[str, Any], llm: ChatOllama) -> str:
    if context.get("analytical_failed"):
        # No se pudo resolver la parte analítica (ninguna tool se ejecutó con
        # éxito) -- se devuelve un mensaje honesto en vez de dejar que el LLM
        # improvise una respuesta sin datos reales de por medio.
        return (
            "No he podido resolver la parte analítica de esta pregunta con las "
            "herramientas disponibles, así que no puedo identificar con certeza "
            "el título sobre el que responder. ¿Puedes reformular la pregunta o "
            "ser más específico?"
        )
    analytical_formatted = _format_tool_results(context["analytical_step"])
    narrative_formatted = _format_documents(context["documents"])
    chain = HYBRID_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "question": question,
            "analytical_context": analytical_formatted,
            "narrative_context": narrative_formatted,
        }
    )

# Función principal de respuesta.
def answer(question: str) -> dict[str, Any]:
    """Punto de entrada único: enruta, recupera contexto y sintetiza la respuesta final."""
    context = route(question)
    llm = _get_llm()

    category = context["category"]
    if category == RouteCategory.NARRATIVE:
        final_answer = _synthesize_narrative(question, context, llm)
    elif category == RouteCategory.ANALYTICAL:
        final_answer = _synthesize_analytical(question, context, llm)
    else:
        final_answer = _synthesize_hybrid(question, context, llm)

    return {
        "question": question,
        "answer": final_answer,
        "category": category,
        "routing_reasoning": context.get("routing_reasoning"),
        "raw_context": context,  # útil para debug y para la Fase de evaluación
    }


# Punto de entrada para pruebas locales.
if __name__ == "__main__":
    test_questions = [
        "¿De qué trata Spider-Man: Across the Spider-Verse?",
        "¿Cuál es la película con mayor recaudación del catálogo?",
        "De las películas de Phase 4, la que tuvo mayor presupuesto, ¿de qué trata?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 70}\n{q}\n{'=' * 70}")
        result = answer(q)
        print(f"[{result['category']}]\n")
        print(result["answer"])