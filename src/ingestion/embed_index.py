"""
embed_index.py
 
Genera el índice vectorial Chroma a partir de los documentos cargados por
chunk_loader.py. Usa embeddings locales vía Ollama, en línea con el resto
del stack del proyecto (laila-rag).
 
Requisitos previos:
    - Ollama corriendo en local: `ollama serve`
    - Modelo de embeddings descargado: `ollama pull nomic-embed-text`
 
Uso:
    python src/ingestion/embed_index.py
    python src/ingestion/embed_index.py --rebuild   # borra y regenera el índice
"""

# Importación de librerías.
import argparse
import shutil
import time 
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.ingestion.chunk_loader import load_documents

# Ruta de persistencia del índice Chroma.
PERSIST_DIR = Path(__file__).resolve().parents(2) / "data" / "vectorstore" / "chroma"
# Nombre de la colección en Chroma.
COLLECTION_NAME = "marvel_media"
# Modelo de embeddings a usar.
EMBED_MODEL = "nomic-embed-text"

# Función principal para construir el índice.
def build_index(rebuild: bool = False) -> Chroma:
    """
    Construye el índice vectorial Chroma a partir de los documentos cargados.
    
    Args:
        rebuild (bool): Si True, borra el índice existente y lo regenera.
        
    Returns:
        Chroma: Instancia del vectorstore construido.
    """
    # Si se pide reconstruir y existe el directorio, lo eliminamos.
    if rebuild and PERSIST_DIR.exists():
        print(f'Borrando indice existente en {PERSIST_DIR}...')
        shutil.rmtree(PERSIST_DIR)
    
    print(f'Cargando documentos desde SQLite...')
    documents = load_documents()
    print(f'    {len(documents)} documentos listos para indexar.')
    
    print(f'Inicializando embeddings ({EMBED_MODEL} via Ollama)...')
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    print(f'Generando y persistiendo el índice Chroma (puede tardar unos minutos)...')
    start = time.time()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    elapsed = time.time() - start
    print(f'    Listo. Tiempo transcurrido: {elapsed:.2f} segundos -> {PERSIST_DIR}.')
    
    return vectorstore

# Función de prueba de smoke test.
def smoke_test(vectorstore: Chroma) -> None:
    """Queries de humo para verificar retrieval + filtrado por metadata."""
    print("\n--- Smoke test: búsqueda semántica libre ---")
    results = vectorstore.similarity_search("multiverse and alternate timelines", k=3)
    for doc in results:
        print(f"  - {doc.metadata['title']} ({doc.metadata['year']})")
 
    print("\n--- Smoke test: búsqueda con filtro de metadata (solo Phase 4) ---")
    results = vectorstore.similarity_search(
        "superhero team fighting a cosmic threat",
        k=3,
        filter={"mcu_phase": "Phase 4"},
    )
    for doc in results:
        print(f"  - {doc.metadata['title']} ({doc.metadata['year']}) - {doc.metadata['mcu_phase']}")    
        

# Ejecución principal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Borra y regenera el índice desde cero")
    args = parser.parse_args()
 
    vs = build_index(rebuild=args.rebuild)
    smoke_test(vs)