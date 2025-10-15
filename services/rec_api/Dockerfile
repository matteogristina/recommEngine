# Use a lightweight Python base image.
FROM python:3.11-slim

# Install system dependencies needed for compiling C extensions (like LightFM).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file and install the dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script into the container.
COPY rec_api.py ./

# Set the entrypoint to run the FastAPI server (uvicorn)
# It will run in the foreground to keep the container alive.
CMD ["uvicorn", "rec_api:app", "--host", "0.0.0.0", "--port", "8001"]