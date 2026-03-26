########################################################
# ERP Agentic Chatbot — Multi-stage Dockerfile
#
# Stage 1 (builder): cài dependencies, tránh để pip cache vào image cuối
# Stage 2 (runtime): image nhẹ, chỉ chứa code + packages đã build
#
# Vì project dùng model AI (embedding, reranker, OCR):
# - Base image: python:3.11-slim (nhỏ hơn full ~600MB)
# - Tách ingestion service riêng nếu cần GPU (xem compose)
########################################################

# ── Stage 1: Builder ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps cho build (gcc cần cho một số packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước → Docker cache layer nếu deps không đổi
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="your-team"
LABEL description="ERP Agentic Chatbot — FastAPI + ReAct Agent"

WORKDIR /app

# System runtime deps (libpq cho psycopg2, curl cho healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages từ builder stage
COPY --from=builder /install /usr/local

# Copy source code
COPY . .

# Non-root user (security best practice)
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && chown -R appuser:appuser /app

# Tạo thư mục cho temp files (upload processing)
RUN mkdir -p /tmp/ingestion && chown appuser:appuser /tmp/ingestion

# HuggingFace cache — model sẽ download vào đây
# Mount volume để persist qua restarts
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME && chown -R appuser:appuser $HF_HOME

USER appuser

# Port mặc định
EXPOSE 8000

# Healthcheck — gọi /health mỗi 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — production dùng gunicorn + uvicorn workers
# Dev override trong compose.yml
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]