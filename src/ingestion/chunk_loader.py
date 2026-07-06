"""
chunk_loader.py
 
Carga los documentos de `rag_documents` (SQLite) como objetos `Document` de
LangChain, enriquecidos con metadata proveniente de `media` para permitir
filtrado en el retriever (por fase MCU, tipo, canon, universo, año).
 
Decisión de diseño:
- Cada fila de `rag_documents` ya es un documento autocontenido por título
  (ver EDA: secciones Title/Plot/Cast/Ratings/Box Office...), así que NO se
  aplica text-splitting adicional por tamaño de token. El chunking natural
  por entidad (una película/serie = un documento) da mejor recall en este
  dataset que trocear por longitud fija, precisamente porque las preguntas
  del asistente son casi siempre sobre una entidad concreta.
- Si en la Fase de evaluación algún documento resulta demasiado largo para
  el modelo de embeddings, se puede introducir un splitter secundario aquí
  mismo (ver TODO al final), sin tocar el resto del pipeline.
 
Uso:
    from src.ingestion.chunk_loader import load_documents
    docs = load_documents()
"""

# Carga de librerías.
from pathlib import Path
import sqlite3
from langchain_core.documents import Document

# Ruta a la base de datos.
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "marvel.db"

# Columnas de media que se propagan como metadata filtrable en el retriever.
METADATA_COLUMNS = [
    "title",
    "year",
    "type",
    "mcu_phase",
    "is_mcu_phase",
    "universe"
]

# Carga de documentos.
def load_documents(db_path: Path = DB_PATH) -> list[Document]:
    """Devuelve la lista de documentos listos para indexar en Chroma."""
    # Error si no encuentra la ruta de la base de datos. 
    if not db_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {db_path}. Ejecuta antes el script de generación de base de datos (build_sqlite.py)."
        )
    
    # Query para cargar los documentos.
    query = f"""
    SELECT
        r.media_id,
        r.document, 
        {", ".join(f"m.{c}" for c in METADATA_COLUMNS)}
    FROM rag_documents r 
    JOIN media m ON m.media_id = r.media_id
    """
    
    # Conexión con la base de datos.
    documents: list[Document] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    
    # Procesamiento de los documentos.
    for row in rows:
        metadata = {col: row[col] for col in METADATA_COLUMNS}
        # Chroma no admite None en metadata; se normaliza a valores serializables.
        metadata = {k: (v if v is not None else "unknown") for k, v in metadata.items()}
        metadata["media_id"] = row["media_id"]
        # is_mcu_canon puede venir como 0/1 o True/False según el CSV origen;
        # se homogeniza a bool para que el filtrado en Chroma sea consistente.
        if "is_mcu_canon" in metadata:
            metadata["is_mcu_canon"] = bool(metadata["is_mcu_canon"]) and metadata["is_mcu_canon"] != "unknown"
 
        documents.append(
            Document(page_content=row["document"], metadata=metadata)
        )
 
    return documents

    # TODO (si hiciera falta en el futuro): si algún documento supera el
    # límite de tokens del modelo de embeddings, aplicar aquí un
    # RecursiveCharacterTextSplitter SOLO sobre los documentos que lo superen,
    # propagando el mismo metadata a cada sub-chunk + un campo `chunk_index`.

# Resumen de la carga.
def summary(documents: list[Document]) -> None:
    """Imprime un resumen rápido para verificar la carga (debug/CLI)."""
    print(f'Documentos cargados: {len(documents)}')
    lengths = [len(d.page_content) for d in documents]
    print(f"Longitud media: {sum(lengths) / len(lengths):.0f} caracteres")
    print(f"Longitud máx: {max(lengths)} | mín: {min(lengths)}")
    print("\nEjemplo de metadata:")
    print(documents[0].metadata)
    print("\nEjemplo de contenido (primeros 300 caracteres):")
    print(documents[0].page_content[:300])
    
# Ejecución.
if __name__ == "__main__":
    docs = load_documents()
    summary(docs)