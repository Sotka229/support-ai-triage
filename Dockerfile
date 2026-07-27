# PoC не имеет внешних зависимостей, поэтому образ — это база + исходники.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONPATH=/app/src:/app
WORKDIR /app

COPY pyproject.toml demo.py ./
COPY src ./src
COPY tools ./tools
COPY data ./data
COPY tests ./tests

RUN pip install --no-cache-dir pytest==8.0.0 && python -m pytest -q

CMD ["python", "demo.py"]
