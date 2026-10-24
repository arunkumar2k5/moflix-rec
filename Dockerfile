FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
# CPU-only torch keeps the image to a modest size; the target host has no GPU.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

RUN useradd --system --uid 1000 moflix && mkdir /data && chown moflix /data
USER moflix

ENV MOFLIX_DB=/data/moflix.db
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "app.main"]
