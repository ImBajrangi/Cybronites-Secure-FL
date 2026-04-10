# Stage 1: Build Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm install
COPY dashboard/ ./
RUN npm run build

# Stage 2: Build Backend & Final Image
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for caching
COPY requirements.txt ./requirements.txt
COPY secure_training_platform/requirements.txt ./stp_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r stp_requirements.txt

# Copy backend code structures
COPY Cybronites/ ./Cybronites/
COPY blockchain/ ./blockchain/
COPY security/ ./security/
COPY utils/ ./utils/
COPY core/ ./core/
COPY secure_training_platform/ ./secure_training_platform/
COPY auth_server/ ./auth_server/

# Copy built frontend from Stage 1
# bridge.py checks both /app/static and /app/dist
COPY --from=frontend-builder /app/dashboard/dist ./dist

# Copy the Cloud Deployment Startup script
COPY deployment_hf/start_cloud.sh ./start.sh
RUN chmod +x start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860
ENV PYTHONPATH=/app
ENV GUARDIAN_DB_PATH="/app/Cybronites/guardian.db"

# Grant permission to Hugging Face user (1000) to write SQLite DB
RUN chown -R 1000:1000 /app

# Hugging Face Spaces use port 7860 by default
EXPOSE 7860

# Use a startup script to run multi-process orchestrator
CMD ["./start.sh"]
