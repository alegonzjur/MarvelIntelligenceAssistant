"""
build_sqlite.py
 
Consolida los 4 CSV crudos del dataset MCU & Marvel Media en una base de
datos SQLite normalizada, lista para el SQL agent y para el pipeline RAG.
 
Decisiones de diseño (documentadas también en el README):
- Clave de unión validada: (title, year) es única en marvel_master.csv y
  cubre el 100% de las claves presentes en ratings, cast y rag.
- Se usa `master.id` como surrogate key (media_id) propagada a las tablas
  hijas, en vez de mantener el JOIN por (title, year) en tiempo de consulta.
  Esto simplifica el SQL que tendrá que generar el agente y evita errores
  de JOIN multi-columna.
- La tabla `media` (antes marvel_master) es la tabla de hechos central.
- `ratings` y `cast` quedan en formato largo (long format), tal y como
  vienen en origen.
- `rag_documents` se mantiene aparte porque no se usa desde el SQL agent,
  solo desde el pipeline de ingestión vectorial (Fase 1).
 
Uso:
    python src/data_prep/build_sqlite.py
"""
# Importación de librerías.
from pathlib import Path 
import sqlite3 
import sys 
import pandas as pd

# Directorios por defecto.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "marvel.db"
MASTER_CSV = RAW_DIR / "marvel_master.csv"
RAG_CSV = RAW_DIR / "marvel_rag.csv"
RATINGS_CSV = RAW_DIR / "marvel_ratings.csv"
CAST_CSV = RAW_DIR / "marvel_cast.csv"


# Función de validación de join keys.
def validate_join_keys(master: df.DataFrame, child: pd.DataFrame, name: str) -> None:
    """
    Valida que las claves de unión (title, year) sean únicas en master
    y que todas las claves de child estén en master.
    """
    master_keys = set(zip(master["title"], master["year"]))
    child_keys = set(zip(child["title"], child["year"]))
    
    # Verificación que las claves childs están en master.
    if orphans:
        raise ValueError(
            f"[{name}] {len(orphans)} claves (title, year) sin match en master: "
            f"{list(orphans)[:5]}..."
        )
    print(f"    OK  {name}: {len(child_keys)} claves, 100% cubiertas por master")


# Función principal de construcción.
def build() -> None:
    if not master_files_exists():
        sys.exit(1)
    
    print("Cargando CSVs...")
    master = pd.read_csv(MASTER_CSV)
    rag = pd.read_csv(RAG_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    cast = pd.read_csv(CAST_CSV)    
    
    if not master["id"].is_unique:
        raise ValueError("master.id no es único, no se puede usar como surrogate key")
    if master.duplicated(subset=["title", "year"]).any():
        raise ValueError("(title, year) duplicado en master, la clave de unión no es válida")
    
    print("Validando integridad referencial (title,year)...")
    validate_join_keys(master, ratings, "ratings")
    validate_join_keys(master, cast, "cast")
    validate_join_keys(master, rag, "rag")
    
    # Mapa (title, year) -> media_id, para propagar el surrogate key
    key_to_id = master.set_index(["title", "year"])["id"].to_dict()
    
    ratings = ratings.copy()
    ratings["media_id"] = ratings.apply(lambda r: key_to_id[(r["title"], r["year"])], axis=1)
    
    cast = cast.copy()
    cast["media_id"] = cast.apply(lambda c: key_to_id[(c["title"], c["year"])], axis=1)
    
    rag = rag.copy()
    rag["media_id"] = rag.apply(lambda r: key_to_id[(r["title"], r["year"])], axis=1)
    
    # Renombrar la tabla de hechos a 'media' (más claro que "master" para el agente SQL)
    media = master.rename(columns={"id": "media_id"})
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    print(f"Creando base de datos SQLite: ´{DB_PATH}...")
    with sqlite3.connect(DB_PATH) as conn:
        media.to_sql("media", conn, index=False, if_exists="replace")
        ratings.drop(columns=["title", "year"]).to_sql(
            "ratings", conn, index=False, if_exists="replace"
        )
        cast.drop(columns=["title", "year"]).to_sql(
            "cast_members", conn, index=False, if_exists="replace"
        )
        rag[["media_id", "document"]].to_sql(
            "rag_documents", conn, index=False, if_exists="replace"
        )
 
        conn.execute("CREATE INDEX idx_ratings_media_id ON ratings(media_id)")
        conn.execute("CREATE INDEX idx_cast_media_id ON cast_members(media_id)")
        conn.execute("CREATE INDEX idx_rag_media_id ON rag_documents(media_id)")
 
        # Usuario de solo lectura a nivel de aplicación (ver Fase de SQL agent):
        # SQLite no soporta roles/permisos nativos, así que la restricción de
        # solo-lectura se aplica en la capa de conexión del agente (Fase 3),
        # no aquí.
 
    print("Listo. Tablas creadas: media, ratings, cast_members, rag_documents")
    print_summary()


# Verifica que los archivos maestros existan.
def master_files_exists() -> bool:
    missing = [p for p in [MASTER_CSV, RAG_CSV, RATINGS_CSV, CAST_CSV] if not p.exists()]
    if missing:
        print(f"Faltan ficheros en data/raw/:")
        for m in missing:
            print(f"    - {m}")
        return False
    return True

# Imprime un resumen de las tablas creadas.
def print_summary() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        for table in ["media", "ratings", "cast_members", "rag_documents"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:15s} {n:5d} filas")
            

if __name__ == "__main__":
    build()