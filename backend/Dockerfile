FROM python:3.11-slim AS development

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# GeoPandas / Fiona / pyogrio need GDAL + GEOS + spatialindex.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libspatialindex-dev \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
