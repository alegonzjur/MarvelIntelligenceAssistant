"""
vector_retriever.py

Envuelve el índice Chroma en una interfaz simple para el router (Fase 3):
búsqueda semántica con filtrado opcional por metadata (mcu_phase, type,
is_mcu_canon, universe, year).

Uso:
    from src.retrieval.vector_retriever import MarvelRetriever

    retriever = MarvelRetriever()
    docs = retriever.search("what happened to Tony Stark", k=3)
    docs = retriever.search("team movies", k=3, mcu_phase="Phase 4")
"""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL

PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "vectorstore" / "chroma"
COLLECTION_NAME = "marvel_media"

# Campos de metadata que se pueden usar como filtro exacto.
FILTERABLE_FIELDS = {"mcu_phase", "type", "is_mcu_canon", "universe", "year", "title"}


class MarvelRetriever:
    def __init__(
        self,
        persist_dir: Path = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"No existe el índice en {persist_dir}. "
                "Ejecuta antes: python -m src.ingestion.embed_index --rebuild"
            )
        self._embeddings = OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_BASE_URL)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(persist_dir),
        )

    def search(
        self,
        query: str,
        k: int = 4,
        **metadata_filters: Any,
    ) -> list[Document]:
        """
        Búsqueda semántica con filtros opcionales de metadata exacta.
        Ejemplo: retriever.search("cosmic threat", k=3, mcu_phase="Phase 4")
        """
        unknown = set(metadata_filters) - FILTERABLE_FIELDS
        if unknown:
            raise ValueError(
                f"Filtro(s) no soportado(s): {unknown}. "
                f"Campos válidos: {FILTERABLE_FIELDS}"
            )

        chroma_filter = self._build_filter(metadata_filters)
        return self._store.similarity_search(query, k=k, filter=chroma_filter)

    def search_with_scores(
        self, query: str, k: int = 4, **metadata_filters: Any
    ) -> list[tuple[Document, float]]:
        """Igual que search(), pero devuelve también la distancia/score —
        útil para la Fase de evaluación (umbral de confianza del retriever)."""
        chroma_filter = self._build_filter(metadata_filters)
        return self._store.similarity_search_with_score(query, k=k, filter=chroma_filter)

    @staticmethod
    def _build_filter(metadata_filters: dict[str, Any]) -> dict | None:
        if not metadata_filters:
            return None
        if len(metadata_filters) == 1:
            key, value = next(iter(metadata_filters.items()))
            return {key: value}
        # Chroma requiere el operador $and explícito para múltiples condiciones.
        return {"$and": [{k: v} for k, v in metadata_filters.items()]}


if __name__ == "__main__":
    retriever = MarvelRetriever()

    print("--- Búsqueda libre ---")
    for doc in retriever.search("betrayal within a team of heroes", k=3):
        print(f"  {doc.metadata['title']} ({doc.metadata['year']})")

    print("\n--- Búsqueda filtrada: solo series, Phase 4 ---")
    for doc in retriever.search(
        "conflict and identity", k=3, mcu_phase="Phase 4", type="series"
    ):
        print(f"  {doc.metadata['title']} ({doc.metadata['year']}) - {doc.metadata['type']}")