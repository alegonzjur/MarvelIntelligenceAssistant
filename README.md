# Marvel Intelligence Assistant

Asistente conversacional híbrido **RAG + SQL agent** sobre el catálogo MCU/Marvel, capaz de responder tanto preguntas narrativas ("¿de qué trata X?") como analíticas ("¿cuál es la película con más recaudación?") y preguntas híbridas que requieren encadenar ambas ("de las películas de Phase 4 con más presupuesto, ¿de qué trata la más cara?").

Dataset base: [MCU and Marvel Media Complete Dataset](https://www.kaggle.com/datasets/vanshkumar007/mcu-and-marvel-media-complete-dataset) (Kaggle).

> **Estado del proyecto:** en desarrollo activo. Este README se actualiza a medida que avanzan las fases. La sección [Estado actual](#estado-actual-de-las-fases) indica qué está hecho, probado, y qué queda pendiente.

---

## Por qué este proyecto

La mayoría de proyectos RAG son "sube un PDF, pregunta sobre el PDF". Este va un paso más allá: combina **recuperación semántica** sobre texto narrativo con una **capa analítica sobre datos estructurados**, y un **router** que decide en tiempo real qué camino tomar — o si hace falta encadenar los dos. Es el patrón de *agentic RAG* que se pide en roles de Data Scientist/GenAI más allá del RAG básico.

---

## Arquitectura

```
Pregunta del usuario
        │
        ▼
   router.py (clasificador LLM)
        │
   ┌────┼─────────────┐
   ▼    ▼              ▼
narrative  analytical  hybrid
   │        │              │
   ▼        ▼              ▼
Chroma    SQL tools    SQL tools → extracción de título → Chroma
(RAG)   (predefinidas  (paso 1: identifica el título)
        + fallback     (paso 2: recupera su narrativa)
        con guardrails)
        │
        ▼
   chain.py (síntesis de la respuesta final) — próxima fase
```

**Dos fuentes de datos, una identidad compartida:** todo el catálogo vive en `data/processed/marvel.db` (SQLite), con `media_id` como clave surrogate que conecta la tabla de hechos (`media`) con sus tablas hijas (`ratings`, `cast_members`, `rag_documents`). El índice vectorial Chroma se construye a partir de `rag_documents`, no del CSV crudo, para tener una única fuente de verdad.

---

## Stack

| Componente | Elección |
|---|---|
| LLM (routing, extracción, síntesis) | Ollama local (`llama3.1` recomendado; `llama3.2:3b` como alternativa ligera) |
| Embeddings | `nomic-embed-text` (Ollama) |
| Vector store | ChromaDB, persistente, con filtrado por metadata |
| Orquestación | LangChain (tool calling + structured output) |
| Capa tabular | SQLite |
| API (pendiente) | FastAPI |
| Frontend (pendiente) | Streamlit |
| Tracking (pendiente) | MLflow |
| Tests (pendiente) | pytest |
| Contenedores (pendiente) | Docker |

---

## Estructura del proyecto

```
marvel-rag-hybrid/
├── data/
│   ├── raw/                      # 4 CSV originales de Kaggle
│   ├── processed/marvel.db       # SQLite consolidado
│   └── vectorstore/chroma/       # índice vectorial persistente
├── src/
│   ├── data_prep/
│   │   └── build_sqlite.py       # consolidación + validación de integridad
│   ├── ingestion/
│   │   ├── chunk_loader.py       # SQLite -> Document (LangChain)
│   │   └── embed_index.py        # genera y persiste el índice Chroma
│   ├── retrieval/
│   │   ├── vector_retriever.py   # búsqueda semántica + filtrado por metadata
│   │   ├── sql_tools.py          # funciones analíticas predefinidas (camino seguro)
│   │   ├── sql_guardrails.py     # validación de SQL libre (camino de escape)
│   │   └── sql_agent.py          # expone ambos caminos como tools de LangChain
│   ├── orchestration/
│   │   └── router.py             # clasifica + recupera contexto (narrative/analytical/hybrid)
│   └── evaluation/
│       └── golden_questions.json # 19 preguntas de referencia, verificadas contra marvel.db
├── notebooks/
│   └── 01_eda_exploracion.ipynb
└── requirements.txt
```

---

## Datos

4 tablas relacionadas por `title` + `year` (clave validada como única y sin huérfanas en las 3 tablas hijas):

- **`marvel_master.csv`** → 161 títulos, 50 columnas (fechas, fase MCU, presupuesto, recaudación, ratings agregados...)
- **`marvel_rag.csv`** → 161 documentos de texto pre-formateados (uno por título), fuente del índice vectorial
- **`marvel_ratings.csv`** → 554 filas, formato largo (una fila por título+fuente de rating)
- **`marvel_cast.csv`** → 1628 filas, formato largo (una fila por título+actor, máx. 15 por título)

### Decisiones de diseño sobre los datos

- **Clave de unión `(title, year)`**, no `tmdb_id`: `tmdb_id` tiene 44 nulos y no es único incluso donde existe. `(title, year)` sí es única en `master` y cubre el 100% de las claves en las 3 tablas hijas (verificado, ver `notebooks/01_eda_exploracion.ipynb`).
- **`media_id` como surrogate key**: en vez de mantener `(title, year)` como *foreign key* de texto en las tablas hijas, se propaga `master.id` (renombrado a `media_id`) a `ratings`, `cast_members` y `rag_documents` en la carga (`build_sqlite.py`). Simplifica los JOINs que tendrá que generar el SQL agent y evita errores de comparación multi-columna.
- **Chunking por entidad, no por tamaño de token**: cada fila de `rag_documents` es un documento autocontenido por título (longitud máx. 1142 caracteres, media 764). No se aplica text-splitting adicional porque las preguntas del asistente son casi siempre sobre una entidad concreta, y trocear por tamaño fijo habría roto esa correspondencia 1:1 entre documento y película/serie.

### Peculiaridades del dataset que afectan al diseño del agente

- `mcu_phase` agrupa las fases recientes como **`"Phase 5/6"`** (valor único, no separado). Un LLM sin este contexto explícito en el prompt generará consultas con `"Phase 5"` y obtendrá listas vacías silenciosamente — pasó durante el desarrollo (ver [Limitaciones](#limitaciones-y-decisiones-de-diseño-frente-a-errores-reales)).
- `budget_usd`/`revenue_usd` solo están disponibles para ~106 de los 161 títulos.
- No todos los títulos tienen las 4 fuentes de rating (IMDb, TMDB, Rotten Tomatoes, Metacritic); algunas series carecen por completo de rating IMDb.
- El cast está limitado a un máximo de 15 personas por título; 46 títulos no tienen cast registrado (producciones de TV antiguas peor documentadas).

---

## SQL agent: por qué un diseño híbrido de dos niveles

Se evaluaron tres enfoques para la capa analítica:

| Enfoque | Ventaja | Riesgo |
|---|---|---|
| Agente SQL 100% libre | Máxima flexibilidad | El LLM puede generar SQL mal formado, ineficiente o peligroso sin guardrails |
| Solo funciones predefinidas | Predecible, seguro, testeado | No escala a preguntas no anticipadas |
| **Híbrido (elegido)** | Casos frecuentes → funciones testeadas; casos no anticipados → fallback con guardrails | Más código, pero mejor relación seguridad/cobertura |

**Camino rápido (`sql_tools.py`):** 7 funciones predefinidas (`top_by_revenue`, `top_by_budget`, `top_by_roi`, `actor_appearances`, `titles_by_actor`, `best_rated`, `count_titles`), cada una con conexión de solo lectura y `LIMIT` explícito.

**Camino de escape (`sql_guardrails.py`):** cuando ninguna función predefinida cubre la pregunta, se permite SQL libre generado por el LLM, pero validado antes de ejecutar:
1. Conexión SQLite en modo `mode=ro` (solo lectura a nivel de driver, no solo de convención)
2. Solo se permite una única sentencia `SELECT` (bloquea `INSERT`/`UPDATE`/`DELETE`/`DROP`/`ATTACH`/`PRAGMA`/etc. y el apilado de sentencias con `;`)
3. Whitelist de tablas (`media`, `ratings`, `cast_members`)
4. `LIMIT` forzado si el LLM no lo incluye
5. Timeout de ejecución

Probado con 5 casos (2 maliciosos, 3 válidos) — ver bloque `if __name__ == "__main__"` de `sql_guardrails.py`.

---

## Router: clasificación + recuperación de contexto

`router.py` separa deliberadamente **clasificación y recuperación de contexto** de la **generación de la respuesta final** (que irá en `chain.py`, aún no construido). Esto permite evaluar la calidad del *routing* (¿recuperó el contexto correcto?) independientemente de la calidad de *redacción* del LLM — son dos fuentes de error distintas.

- **`classify()`**: clasifica la pregunta en `narrative` / `analytical` / `hybrid` mediante `with_structured_output` (salida forzada a un modelo Pydantic, no parseo de texto libre).
- **`answer_narrative()`**: búsqueda semántica directa vía `MarvelRetriever`.
- **`answer_analytical()`**: tool calling nativo (`bind_tools`) — el LLM elige qué función de `sql_tools.py` invocar; se devuelve el resultado crudo de la tool, no una redacción del LLM.
- **`answer_hybrid()`**: encadena tres pasos — (1) resuelve la parte analítica para identificar un título concreto, (2) extrae `título`+`año` del resultado de la tool con una segunda llamada `with_structured_output`, (3) recupera el documento narrativo filtrando por metadata exacta; si el filtro exacto no encuentra nada, cae a búsqueda semántica libre usando el título extraído como query.

---

## Golden questions

`src/evaluation/golden_questions.json` — 19 preguntas de referencia (6 narrativas, 9 analíticas, 4 híbridas), **con respuestas verificadas contra `marvel.db` real**, no estimadas. Sirven de doble propósito: guiaron el diseño del router (qué casos debía cubrir) y servirán de base para la Fase de evaluación (RAGAS para retrieval narrativo, exactitud de resultado para SQL).

Incluye casos pensados a propósito para estresar el diseño:
- `anl-08`: un parámetro de tool con espacio (`"Rotten Tomatoes"`), para detectar fallos sutiles de paso de argumentos.
- `anl-09`: una agregación (`AVG`) que ninguna función predefinida cubre, para forzar y validar el camino de fallback SQL.
- `hyb-01` a `hyb-04`: encadenamiento analítico → narrativo en orden correcto.

---

## Estado actual de las fases

| Fase | Estado |
|---|---|
| Fase 0 — Consolidación de datos (`build_sqlite.py`) | ✅ Hecho y probado |
| EDA (`01_eda_exploracion.ipynb`) | ✅ Hecho |
| Fase 1 — Ingestión vectorial (`chunk_loader.py`, `embed_index.py`) | ✅ Hecho y probado en local con Ollama |
| Fase 2 — Capa SQL (`sql_tools.py`, `sql_guardrails.py`, `sql_agent.py`) | ✅ Hecho y probado |
| Golden questions | ✅ Hecho |
| Fase 3 — Router (`router.py`) | ✅ Hecho y probado en local (3/3 casos correctos con `llama3.2:3b`) |
| Fase 3 — `chain.py` (síntesis de respuesta final) | ⬜ Pendiente |
| Fase 4 — Evaluación sistemática (RAGAS + exactitud SQL sobre las 19 golden questions) | ⬜ Pendiente |
| Fase 5 — API (FastAPI) + Frontend (Streamlit) | ⬜ Pendiente |
| Fase 6 — MLOps ligero (MLflow, pytest, DVC opcional) | ⬜ Pendiente |

---

## Limitaciones y decisiones de diseño frente a errores reales

Documentado deliberadamente, no maquillado — es información útil tanto para seguir desarrollando el proyecto como para explicar decisiones en una entrevista técnica.

- **Modelo pequeño (`llama3.2:3b`) vs. tool calling fiable.** Las pruebas locales se hicieron con `llama3.2:3b` por limitaciones de hardware. Se observó que el modelo pasa correctamente los parámetros a las tools pero a veces como tipo incorrecto (ej. `n: "1"` en vez de `n: 1`) — no falla porque Pydantic (vía `StructuredTool.from_function`) coacciona el tipo automáticamente, pero es una señal de que con un modelo más grande (`llama3.1`) esto sería más fiable. Pendiente de reproducir con `llama3.1` en otro equipo.
- **`top_by_budget` no existía en el primer diseño.** El set de golden questions incluía una pregunta híbrida sobre "película con más presupuesto de Phase 4" asumiendo que caería al fallback SQL con guardrails. En la práctica, con el modelo pequeño, confiar en que el LLM genere SQL libre correcto es menos fiable que ampliar el catálogo de funciones predefinidas. Se añadió `top_by_budget` como función explícita tras detectar el fallo. **Lección de diseño:** cuanto más pequeño el modelo, más conviene invertir en funciones predefinidas y reducir la dependencia del camino de escape.
- **Extracción de título en `answer_hybrid()` no siempre coincide con el filtro exacto.** En una prueba real, el LLM extrajo el título traducido al español ("Doctor Strange en el multiverso de la locura") en vez del título original en inglés de la metadata. El filtro exacto por `(title, year)` falló silenciosamente, pero el sistema cayó correctamente al fallback de búsqueda semántica libre, que sí recuperó el documento correcto. El diseño de doble capa (filtro exacto → fallback semántico) demostró ser necesario, no redundante.
- **Ambigüedad de fase MCU (`"Phase 5/6"`).** Sin indicárselo explícitamente en el prompt/schema, un LLM asume que existen "Phase 5" y "Phase 6" por separado y genera consultas que devuelven listas vacías sin avisar. Documentado en el prompt del SQL agent (`SCHEMA_DESCRIPTION` en `sql_agent.py`) y en los golden questions.
- **Cobertura de tools de solo lectura + guardrails, no permisos a nivel de SQLite.** SQLite no soporta roles/usuarios nativos, así que la restricción de solo lectura se aplica en la capa de conexión (`mode=ro`) y en la validación de la query (`sql_guardrails.py`), no en la base de datos en sí.

---

## Cómo ejecutar lo que hay hasta ahora

```bash
conda activate marvel-rag
pip install -r requirements.txt

# Fase 0: consolidar datos
python src/data_prep/build_sqlite.py

# Fase 1: indexar (requiere Ollama corriendo + `ollama pull nomic-embed-text`)
python -m src.ingestion.embed_index --rebuild

# Fase 2: probar la capa SQL
python -m src.retrieval.sql_tools
python -m src.retrieval.sql_guardrails
python -m src.retrieval.sql_agent

# Fase 3: probar el router (requiere `ollama pull llama3.1` o `llama3.2:3b`)
python -m src.orchestration.router
```

---

## Próximos pasos

1. `chain.py` — síntesis de la respuesta final en lenguaje natural a partir del contexto que devuelve `router.py`
2. Evaluación sistemática sobre las 19 golden questions (RAGAS + exactitud SQL)
3. API FastAPI + frontend Streamlit
4. MLflow para trackear variantes de prompt/chunking/modelo; pytest para router, tools y retriever