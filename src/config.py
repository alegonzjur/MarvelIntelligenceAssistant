"""
config.py

Configuración centralizada vía variables de entorno. Existe principalmente
por Docker: dentro de un contenedor, "localhost:11434" apunta al propio
contenedor, no al Ollama que corre en la máquina host -- así que la URL
tiene que ser configurable (host.docker.internal, la IP del host, etc.)
en vez de estar hardcodeada como antes en cada módulo.

Fuera de Docker, los valores por defecto son los mismos que se han usado
durante todo el desarrollo, así que no cambia nada si no se define ninguna
variable de entorno.
"""

import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "llama3.2:3b")
SYNTHESIS_MODEL = os.environ.get("SYNTHESIS_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")