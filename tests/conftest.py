"""
conftest.py

Fixtures y configuración compartida de pytest.

Decisión de diseño: los tests que dependen de marvel.db o del índice Chroma
se SALTAN automáticamente (no fallan) si esos artefactos no existen en el
entorno donde corre pytest -- por ejemplo, en un CI que no tenga los datos
descargados. Esto se hace con `pytest.mark.skipif`, no ignorando el test
sin más, para que quede visible en el resumen de pytest que algo no se
pudo comprobar, en vez de dar una falsa sensación de "todo verde".
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "marvel.db"
VECTORSTORE_PATH = PROJECT_ROOT / "data" / "vectorstore" / "chroma"

requires_db = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="marvel.db no encontrado -- ejecuta antes: python src/data_prep/build_sqlite.py",
)

requires_vectorstore = pytest.mark.skipif(
    not VECTORSTORE_PATH.exists(),
    reason="Índice Chroma no encontrado -- ejecuta antes: python -m src.ingestion.embed_index --rebuild",
)