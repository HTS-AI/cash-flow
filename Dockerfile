## --------------------
## Build frontend (React / Vite)
## --------------------
FROM node:20-alpine AS frontend-build
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


## --------------------
## Runtime (FastAPI + built UI)
## --------------------
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code + assets
COPY . .

# Copy built frontend into path expected by FastAPI
COPY --from=frontend-build /frontend/dist ./frontend_dist

# Make start script executable
RUN chmod +x ./start.sh

# Cloud Run port
EXPOSE 8080

# Start app
CMD ["sh", "./start.sh"]