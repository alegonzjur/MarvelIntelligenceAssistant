"""
sql_guardrails.py
 
Guardrails para el "camino de escape" del SQL agent: cuando una pregunta
no encaja en ninguna función predefinida de sql_tools.py, se le permite al
LLM generar SQL libre, pero SIEMPRE pasando por estas validaciones antes
de ejecutar nada.
 
Guardrails aplicados:
1. Conexión SQLite en modo solo lectura (falla a nivel de driver, no solo
   de convención en el código).
2. Solo se permite una única sentencia que empiece por SELECT (bloquea
   INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA/etc., y bloquea el apilado de
   sentencias con ';').
3. Whitelist de tablas: solo puede referenciar media, ratings, cast_members.
4. LIMIT forzado si el LLM no lo incluye, para no volcar cientos de filas
   al contexto del LLM final.
5. Timeout de ejecución.
"""
# Importación de librerías
import re 
import sqlite3
from pathlib import Path

# Ruta al directorio de DB.
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" /"marvel.db"
# Whitelist de tablas permitidas
ALLOWED_TABLES = {"media", "ratings", "cast_members"}
# Limit por defecto
DEFAULT_LIMIT = 25
# Timeout de ejecución
TIMEOUT_SECONDS = 5
# Keywords prohibidos para operaciones de escritura.
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|VACUUM|REPLACE|CREATE|TRIGGER)\b",
    re.IGNORECASE,
)

class UnsafeQueryError(Exception):
    pass

# Función para validar la consulta SQL
def validate_query(sql: str) -> str:
    """Lanza UnsafeQueryError si el SQL no cumple los guardrails.
    Devuelve el SQL (con LIMIT añadido si hacía falta) si es válido."""
    sql = sql.strip().rstrip(";")
    
    if ";" in sql:
        raise UnsafeQueryError("No se permite apilar varias sentencias SQL.")
 
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise UnsafeQueryError("Solo se permiten sentencias SELECT.")
 
    if _FORBIDDEN_KEYWORDS.search(sql):
        raise UnsafeQueryError("La consulta contiene una palabra clave no permitida.")
    # Verificar tablas referenciadas
    referenced = set(re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", sql, re.IGNORECASE))
    referenced_tables = {t for pair in referenced for t in pair if t}
    unknown_tables = referenced_tables - ALLOWED_TABLES
    if unknown_tables:
        raise UnsafeQueryError(f"Tabla(s) no permitida(s): {unknown_tables}")
 
    if not re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
        sql = f"{sql} LIMIT {DEFAULT_LIMIT}"
 
    return sql

# Función para ejecutar consultas SQL de forma segura
def run_safe_query(sql: str, db_path: Path = DB_PATH) -> str:
    """Valida y ejecuta una query SQL libre, con timeout y conexión de solo lectura."""
    try:
        safe_sql = validate_query(sql)
    except UnsafeQueryError as e:
        return f"Consulta rechazada por seguridad: {e}"
 
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=TIMEOUT_SECONDS)
        conn.execute(f"PRAGMA busy_timeout = {TIMEOUT_SECONDS * 1000}")
        cursor = conn.execute(safe_sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        return f"Error ejecutando la consulta: {e}"
 
    if not rows:
        return "La consulta no devolvió resultados."
 
    header = " | ".join(columns)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
    return f"{header}\n{body}"
 
# Función principal para ejecutar consultas SQL de forma segura
if __name__ == "__main__":
    tests = [
        "SELECT title, year FROM media WHERE mcu_phase = 'Phase 5' ORDER BY year",
        "DROP TABLE media",
        "SELECT * FROM media; DELETE FROM media",
        "SELECT * FROM sqlite_master",
        "SELECT title FROM media LIMIT 3",  # ya trae LIMIT, no debe duplicarse
    ]
    for t in tests:
        print(f"\n>>> {t}")
        print(run_safe_query(t))