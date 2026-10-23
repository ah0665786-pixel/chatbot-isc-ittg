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
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# El modelo de embeddings se descarga al construir la imagen: así el bot y el
# panel arrancan sin internet hacia Hugging Face y no compiten por descargarlo.
ARG EMBED_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"
# Ya está en caché: en ejecución no consultar el Hub (arranca aunque no haya internet).
ENV HF_HUB_OFFLINE=1

COPY app ./app

# data/ (documentos, índice FAISS, SQLite) se monta como volumen desde compose.
CMD ["python", "-m", "app.telegram_bot"]
