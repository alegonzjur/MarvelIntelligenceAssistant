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

import argparse
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL
from src.ingestion.chunk_loader import load_documents

PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "vectorstore" / "chroma"
COLLECTION_NAME = "marvel_media"


def build_index(rebuild: bool = False) -> Chroma:
    if rebuild and PERSIST_DIR.exists():
        print(f"Borrando índice existente en {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando documentos desde SQLite...")
    documents = load_documents()
    print(f"  {len(documents)} documentos listos para indexar")

    print(f"Inicializando embeddings ({EMBEDDING_MODEL} vía Ollama)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    print("Generando y persistiendo el índice Chroma (puede tardar unos minutos)...")
    start = time.time()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    elapsed = time.time() - start
    print(f"Índice creado en {elapsed:.1f}s -> {PERSIST_DIR}")

    return vectorstore


def smoke_test(vectorstore: Chroma) -> None:
    """Un par de queries de humo para verificar retrieval + filtrado por metadata."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Borra y regenera el índice desde cero")
    args = parser.parse_args()

    vs = build_index(rebuild=args.rebuild)
    smoke_test(vs)