FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first, so sentence-transformers does not pull the multi-gigabyte CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml ./
COPY app ./app
RUN pip install .

RUN useradd --create-home --uid 1000 moflix \
    && mkdir /data \
    && chown moflix:moflix /data
USER moflix

ENV MOFLIX_DB=/data/moflix.db
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--log-config", "app/log_config.json", "--no-access-log"]
