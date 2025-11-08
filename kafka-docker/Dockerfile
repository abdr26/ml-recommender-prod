#  Use a lightweight official Python base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies (for numpy/pandas/pyarrow builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose FastAPI port (Azure expects 8080)
EXPOSE 8080

# Default startup command for Azure Container Apps
CMD ["uvicorn", "recommender.api:app", "--host", "0.0.0.0", "--port", "8080"]
