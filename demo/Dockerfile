FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RIVER_MODEL_PATH=/app/model OMP_NUM_THREADS=2
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . 'fastapi==0.141.1' 'uvicorn==0.53.0'
COPY demo ./demo
COPY model ./model
CMD ["sh", "-c", "exec uvicorn demo.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
