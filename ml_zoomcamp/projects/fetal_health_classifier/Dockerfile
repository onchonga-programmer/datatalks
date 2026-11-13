FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY models/model.bin ./models/model.bin

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "src", "predict:app", "--host", "0.0.0.0", "--port", "8000"]