ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app \
    CONFIG_PATH=/app/config.ini \
    MODEL_PATH=/app/models/model.joblib \
    ANSIBLE_HOME=/tmp/ansible-home \
    ANSIBLE_LOCAL_TEMP=/tmp/ansible-local

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install --no-deps .

COPY config.ini ansible.cfg ./
COPY models/model.joblib ./models/model.joblib

RUN chown -R app:app /app
USER app

FROM base as test

USER root
COPY requirements-test.txt ./requirements-test.txt
RUN python -m pip install -r requirements-test.txt
COPY tests ./tests
RUN chown -R app:app /app
USER app

CMD ["pytest", "-m", "unit", "--cov", "--cov-report=term-missing"]

FROM base AS runtime

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "fashion_mnist_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
