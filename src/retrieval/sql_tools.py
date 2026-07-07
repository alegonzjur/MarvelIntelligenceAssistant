"""
sql_tools.py
 
Camino "rápido y seguro" del SQL agent híbrido: funciones predefinidas que
cubren las preguntas analíticas más frecuentes sobre el dataset, sin dejar
que el LLM genere SQL libre para estos casos.
 
Cada función:
- Usa una conexión de solo lectura a SQLite.
- Aplica un LIMIT explícito.
- Devuelve resultados ya formateados en texto, listos para el LLM final.
 
Si una pregunta no encaja en ninguna de estas funciones, el router (Fase 3)
debe caer al agente SQL con guardrails (ver sql_agent.py).
"""

# Importación de librerías
from pathlib import Path
import sqlite3

# Ruta a la base de datos
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "marvel.db"

# Función para crear conexión de solo lectura
def _read_only_connection(
    db_path: Path = DB_PATH
) -> sqlite3.Connection:
    """Conexión en modo solo lectura: cualquier intento de escritura falla
    a nivel de SQLite, no solo por convención en el código."""
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)

# Función para obtener los top N títulos por recaudación
def top_by_revenue(mcu_phase: str | None = None, n: int = 5) -> str:
    """Top N títulos por recaudación (revenue_usd), opcionalmente filtrado por fase."""
    # Construir cláusula WHERE y parámetros
    where = "WHERE revenue_usd IS NOT NULL"
    params: list = []
    if mcu_phase: 
        where += " AND mcu_phase = ?"
        params.append(mcu_phase)
    # Construir consulta SQL
    query = f"""
    SELECT title, year, mcu_phase, revenue_usd
    FROM media
    {where}
    ORDER BY revenue_usd DESC
    LIMIT ?
    """
    # Agregar parámetro para LIMIT
    params.append(n)
    with _read_only_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    if not rows:
        return "No se encontraron resultados para ese filtro."
    # Formatear resultados
    lines = [f"{i+1}. {t} ({y}, {p}): ${r:,.0f}" for i, (t, y, p, r) in enumerate(rows)]
    return "\n".join(lines)

# Función para obtener los top N títulos por presupuesto
def top_by_budget(mcu_phase: str | None = None, n: int = 5) -> str:
    """Top N títulos por presupuesto (budget_usd), opcionalmente filtrado por fase."""
    where = "WHERE budget_usd IS NOT NULL"
    params: list = []
    if mcu_phase:
        where += " AND mcu_phase = ?"
        params.append(mcu_phase)
 
    query = f"""
        SELECT title, year, mcu_phase, budget_usd
        FROM media
        {where}
        ORDER BY budget_usd DESC
        LIMIT ?
    """
    params.append(n)
 
    with _read_only_connection() as conn:
        rows = conn.execute(query, params).fetchall()
 
    if not rows:
        return "No se encontraron resultados para ese filtro."
 
    lines = [f"{i+1}. {t} ({y}, {p}): ${b:,.0f}" for i, (t, y, p, b) in enumerate(rows)]
    return "\n".join(lines)

# Función para obtener los top N títulos por ROI
def top_by_roi(mcu_phase: str | None = None, n: int = 5) -> str:
    """Top N títulos por ROI (revenue/budget), opcionalmente filtrado por fase."""
    # Construir cláusula WHERE y parámetros
    where = "WHERE revenue_usd IS NOT NULL AND budget_usd IS NOT NULL AND budget_usd > 0"
    params: list = []
    if mcu_phase:
        where += " AND mcu_phase = ?"
        params.append(mcu_phase)
        
    # Construir consulta SQL
    query = f"""
    SELECT title, year, mcu_phase, revenue_usd, budget_usd,
        ROUND(revenue_usd * 1.0 / budget_usd, 2) AS roi
    FROM media 
    {where}
    ORDER BY roi DESC
    LIMIT ?
    """
    params.append(n)
    # Ejecutar consulta
    with _read_only_connection() as conn:
        rows = conn.execute(query, params).fetchall()
 
    if not rows:
        return "No se encontraron resultados para ese filtro."
    
    # Formatear resultados
    lines = [
        f"{i+1}. {t} ({y}, {p}): ROI x{roi} (revenue ${rev:,.0f} / budget ${bud:,.0f})"
        for i, (t, y, p, rev, bud, roi) in enumerate(rows)
    ]
    return "\n".join(lines)

# Función para obtener los top N actores por número de apariciones
def actor_appearances(n: int = 10) -> str:
    """Top N actores por número de apariciones en el catálogo."""
    # Construir consulta SQL
    query = """
        SELECT actor_name, COUNT(*) AS appearances
        FROM cast_members
        GROUP BY actor_name
        ORDER BY appearances DESC
        LIMIT ?
    """
    # Ejecutar consulta
    with _read_only_connection() as conn:
        rows = conn.execute(query, (n,)).fetchall()
 
    # Formatear resultados
    lines = [f"{i+1}. {name}: {count} títulos" for i, (name, count) in enumerate(rows)]
    return "\n".join(lines)


# Función para obtener todos los títulos en los que aparece un actor concreto
def titles_by_actor(actor_name: str) -> str:
    """Todos los títulos en los que aparece un actor concreto (búsqueda parcial, case-insensitive)."""
    # Construir consulta SQL
    query = """
        SELECT m.title, m.year
        FROM cast_members c
        JOIN media m ON m.media_id = c.media_id
        WHERE c.actor_name LIKE ?
        ORDER BY m.year
    """
    # Ejecutar consulta
    with _read_only_connection() as conn:
        rows = conn.execute(query, (f"%{actor_name}%",)).fetchall()
 
    if not rows:
        return f"No se encontró ningún título con un actor que coincida con '{actor_name}'."
 
    lines = [f"- {t} ({y})" for t, y in rows]
    return "\n".join(lines)
 

# Función para obtener los títulos mejor valorados según una fuente de rating concreta
def best_rated(source: str = "IMDb", mcu_phase: str | None = None, n: int = 5) -> str:
    """Top N títulos mejor valorados según una fuente de rating concreta
    (valores válidos en origen: IMDb, TMDB, Rotten Tomatoes, Metacritic;
    la comparación es insensible a mayúsculas), opcionalmente por fase."""
    
    # Construir cláusula WHERE y parámetros
    where = "WHERE r.source = ? COLLATE NOCASE"
    params: list = [source]
    if mcu_phase:
        where += " AND m.mcu_phase = ?"
        params.append(mcu_phase)
    # Construir consulta SQL
    query = f"""
        SELECT m.title, m.year, r.score
        FROM ratings r
        JOIN media m ON m.media_id = r.media_id
        {where}
        ORDER BY r.score DESC
        LIMIT ?
    """
    params.append(n)
 
    # Ejecutar consulta 
    with _read_only_connection() as conn:
        rows = conn.execute(query, params).fetchall()
 
    if not rows:
        return f"No hay ratings de la fuente '{source}' para ese filtro."
 
    lines = [f"{i+1}. {t} ({y}): {score}" for i, (t, y, score) in enumerate(rows)]
    return "\n".join(lines)
 
 
def count_titles(mcu_phase: str | None = None, media_type: str | None = None) -> str:
    """Cuenta títulos, opcionalmente filtrado por fase y/o tipo (movie/series)."""
    # Construir cláusula WHERE y parámetros
    where = "WHERE 1=1"
    params: list = []
    # Agregar filtros opcionales
    if mcu_phase:
        where += " AND mcu_phase = ?"
        params.append(mcu_phase)
    if media_type:
        where += " AND type = ?"
        params.append(media_type)
 
    # Construir consulta SQL
    query = f"SELECT COUNT(*) FROM media {where}"
    
    # Ejecutar consulta
    with _read_only_connection() as conn:
        (count,) = conn.execute(query, params).fetchone()
 
    return str(count)
 

# ==================== Pruebas ====================
if __name__ == "__main__":
    print("=== Top 5 por revenue, Phase 4 ===")
    print(top_by_revenue(mcu_phase="Phase 4", n=5))
 
    print("\n=== Top 5 por ROI, todas las fases ===")
    print(top_by_roi(n=5))
 
    print("\n=== Top 10 actores con más apariciones ===")
    print(actor_appearances(n=10))
 
    print("\n=== Títulos con Robert Downey ===")
    print(titles_by_actor("Robert Downey"))
 
    print("\n=== Mejor valoradas según IMDb ===")
    print(best_rated(source="imdb", n=5))
 
    print("\n=== Conteo: series de Phase 4 ===")
    print(count_titles(mcu_phase="Phase 4", media_type="series"))