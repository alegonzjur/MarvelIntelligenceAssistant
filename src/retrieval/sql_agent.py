"""
sql_agent.py

Expone la capa analítica como tools de LangChain, combinando:
1. Camino rápido: funciones predefinidas de sql_tools.py (preferido siempre
   que la pregunta encaje).
2. Camino de escape: SQL libre generado por el LLM, validado por
   sql_guardrails.py (solo si ninguna función predefinida cubre la pregunta).

El router (Fase 3) decide primero si la pregunta es narrativa (-> retriever
vectorial) o analítica (-> estas tools). Dentro de "analítica", el propio
LLM decide qué tool usar en base a su descripción — por eso las
descripciones son específicas y mencionan cuándo NO usarlas.

Nota sobre nomenclatura real de mcu_phase (importante para el prompt del
agente): 'Phase 1'..'Phase 4', 'Phase 5/6' (agrupadas), 'Non-MCU', 'Pre-MCU'.
No existe 'Phase 5' ni 'Phase 6' por separado.
"""
# Importación de librerías
from langchain_core.tools import StructuredTool
from src.retrieval import sql_tools
from src.retrieval.sql_guardrails import run_safe_query

# Constantes 
VALID_PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5/6", "Non-MCU", "Pre-MCU"]

# Descripción del esquema
SCHEMA_DESCRIPTION = """
Tablas disponibles (solo lectura):
- media(media_id, title, year, type[movie|series], mcu_phase, is_mcu_canon, universe,
        budget_usd, revenue_usd, box_office_usd, runtime_minutes, director, ...)
- ratings(media_id, source[IMDb|TMDB|Rotten Tomatoes|Metacritic], score)
- cast_members(media_id, actor_name, character, cast_order, popularity)

Valores válidos de mcu_phase: {phases}. NO existen 'Phase 5' ni 'Phase 6' por separado,
están agrupadas como 'Phase 5/6'.
""".format(phases=", ".join(VALID_PHASES))

# Funciones
def get_predefined_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=sql_tools.top_by_revenue,
            name="top_by_revenue",
            description=(
                "Devuelve el top N de títulos por recaudación (revenue_usd). "
                "Úsala para preguntas tipo '¿cuál es la película que más ha recaudado?'. "
                f"mcu_phase debe ser uno de: {VALID_PHASES} o None para todas."
            ),
        ),
        StructuredTool.from_function(
            func=sql_tools.top_by_budget,
            name="top_by_budget",
            description=(
                "Devuelve el top N de títulos por presupuesto de producción (budget_usd). "
                "Úsala para preguntas sobre qué película fue más cara de producir, NO para "
                "preguntas sobre recaudación (usa top_by_revenue) ni rentabilidad (usa top_by_roi). "
                f"mcu_phase debe ser uno de: {VALID_PHASES} o None para todas."
            ),
        ),
        StructuredTool.from_function(
            func=sql_tools.top_by_roi,
            name="top_by_roi",
            description=(
                "Devuelve el top N de títulos por ROI (revenue/budget). "
                "Úsala para preguntas sobre rentabilidad, no recaudación bruta. "
                f"mcu_phase debe ser uno de: {VALID_PHASES} o None para todas."
            ),
        ),
        StructuredTool.from_function(
            func=sql_tools.actor_appearances,
            name="actor_appearances",
            description="Devuelve el ranking de actores con más apariciones en el catálogo.",
        ),
        StructuredTool.from_function(
            func=sql_tools.titles_by_actor,
            name="titles_by_actor",
            description=(
                "Devuelve todos los títulos en los que aparece un actor concreto. "
                "actor_name admite coincidencia parcial (ej. 'Downey' encuentra 'Robert Downey Jr.')."
            ),
        ),
        StructuredTool.from_function(
            func=sql_tools.best_rated,
            name="best_rated",
            description=(
                "Devuelve el top N de títulos mejor valorados según una fuente de rating "
                "(IMDb, TMDB, Rotten Tomatoes o Metacritic). "
                f"mcu_phase debe ser uno de: {VALID_PHASES} o None para todas."
            ),
        ),
        StructuredTool.from_function(
            func=sql_tools.count_titles,
            name="count_titles",
            description=(
                "Cuenta cuántos títulos hay, opcionalmente filtrando por fase y/o tipo "
                "(movie/series). Úsala para preguntas tipo '¿cuántas películas tiene Phase 4?'."
            ),
        ),
    ]


# Función de último recurso
def get_fallback_tool() -> StructuredTool:
    """Tool de último recurso: SQL libre con guardrails. El LLM solo debería
    recurrir a esta si ninguna tool predefinida cubre la pregunta."""
    return StructuredTool.from_function(
        func=run_safe_query,
        name="run_safe_sql_query",
        description=(
            "ÚLTIMO RECURSO. Ejecuta una consulta SQL SELECT de solo lectura sobre las "
            "tablas media, ratings y cast_members cuando ninguna otra tool encaja con la "
            "pregunta. Usa esta tool si top_by_revenue, top_by_roi, top_by_budget, "
            "actor_appearances, titles_by_actor, best_rated o count_titles no pueden "
            "responder la pregunta -- EN PARTICULAR para cualquier AGREGACIÓN que no sea "
            "un simple conteo o un top-N: media/promedio (AVG), suma total (SUM), mínimo "
            "(MIN), o cualquier pregunta con las palabras 'media', 'promedio', 'total', "
            "'suma'. Estas agregaciones NO están cubiertas por ninguna tool predefinida, "
            "así que si la pregunta las pide, usa ESTA tool directamente sin probar antes "
            "top_by_revenue/top_by_budget/top_by_roi (esas solo devuelven un ranking, "
            "nunca un promedio). La consulta debe ser un único SELECT; cualquier otra "
            "sentencia será rechazada.\n" + SCHEMA_DESCRIPTION
        ),
    )

# Función principal
def get_all_tools() -> list[StructuredTool]:
    return get_predefined_tools() + [get_fallback_tool()]


# Punto de entrada
if __name__ == "__main__":
    tools = get_all_tools()
    print(f"{len(tools)} tools disponibles para el agente analítico:\n")
    for t in tools:
        print(f"- {t.name}: {t.description.strip().splitlines()[0]}")

    print("\n--- Invocación directa de ejemplo (top_by_roi) ---")
    print(tools[1].invoke({"mcu_phase": "Phase 3", "n": 3}))