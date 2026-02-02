# Use Python 3.11 Linux base
FROM python:3.11-slim

# Install system build tools
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and the pre-built chroma_db folder
COPY . .

# Expose port for Render
EXPOSE 10000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]