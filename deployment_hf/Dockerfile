# Stage 1: Build Frontend (Institutional Dashboard)
FROM node:20-slim AS frontend-builder
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm install
COPY dashboard/ ./
RUN npm run build

# Stage 2: Production Hub
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up environment and user for Hugging Face (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PORT=7860 \
    FLOWER_PORT=8080 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy Requirements and Install (as root for system-wide access)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY --chown=user Cybronites/ $HOME/app/Cybronites/
COPY --chown=user blockchain/ $HOME/app/blockchain/
COPY --chown=user security/ $HOME/app/security/
COPY --chown=user run_local.py $HOME/app/
COPY --chown=user README.md $HOME/app/

# Dashboard Static Assets (from Stage 1)
COPY --chown=user --from=frontend-builder /app/dashboard/dist $HOME/app/dist

# Ensure the app directory is fully owned and writable by the user
RUN chmod -R 777 $HOME/app
USER user

# CMD to start the orchestrator (Unified Entrypoint)
CMD ["python3", "run_local.py"]
