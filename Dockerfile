FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and trained model
COPY challenge/ ./challenge/

# Expose port for FastAPI app
EXPOSE 8000

# App port as env (can be changed on runtime)
ENV PORT=8000

# Run the FastAPI app with uvicorn
CMD exec uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT}