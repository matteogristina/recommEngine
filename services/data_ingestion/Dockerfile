# Use a lightweight Python base image.
FROM python:3.9-slim

# Install the necessary PostgreSQL development headers.
# This provides 'pg_config' and other tools for building psycopg2.
RUN apt-get update && apt-get install -y libpq-dev gcc

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install the dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ingestion script into the container.
COPY ingest.py ./

# Define the command to run when the container starts.
CMD ["python", "ingest.py"]