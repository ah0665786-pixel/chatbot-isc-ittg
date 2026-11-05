# Imagen única para el bot de Telegram y el panel de administración
# (docker-compose.yml elige qué proceso arranca cada servicio).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# PyTorch solo-CPU primero: si sentence-transformers lo instalara desde PyPI
# traería la versión con CUDA (varios GB que el servidor no usa).
# Versiones fijas (2026-11-05): con requirements.txt solo (">=") la imagen traía
# sentence-transformers 6.2, torch 2.15 y transformers 5.22, y la máquina de
# desarrollo 6.1, 2.14 y 5.17 (requirements.lock). Los puntajes de similitud
# cambiaban lo suficiente para mover el orden de los fragmentos: el glosario
# (pág. 183, 70% de créditos del servicio social) quedaba 4.º en desarrollo y
# 7.º en el bot. Deben coincidir con la versión de torch de requirements.lock.
ARG TORCH_VERSION=2.14.0
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION}

# requirements.lock como restricción: fija la versión de todo lo que trae (y lo
# que se instaló en desarrollo); lo que no trae (bot, panel, PostgreSQL) queda
# como lo pide requirements.txt.
COPY requirements.txt requirements.lock ./
RUN pip install -r requirements.txt -c requirements.lock

# El modelo de embeddings se descarga al construir la imagen: así el bot y el
# panel arrancan sin internet hacia Hugging Face y no compiten por descargarlo.
ARG EMBED_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"
# Ya está en caché: en ejecución no consultar el Hub (arranca aunque no haya internet).
ENV HF_HUB_OFFLINE=1

COPY app ./app

# data/ (documentos, índice FAISS, SQLite) se monta como volumen desde compose.
CMD ["python", "-m", "app.telegram_bot"]
