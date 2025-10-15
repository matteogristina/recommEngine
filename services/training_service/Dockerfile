# Use a lightweight Python base image.
FROM python:3.11-slim

# Install system dependencies needed for psycopg2 (libpq-dev) and building packages (gcc).
# This is crucial for installing the 'psycopg2-binary' package.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install the dependencies.
# We do this in a separate layer to leverage Docker's build cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ingestion script into the container.
COPY train_model.py ./

# Create the directory where the final model artifacts will be saved (as defined in train_model.py)
# This is necessary for the volume mount to work correctly.
RUN mkdir -p /app/model_artifacts

# The container is designed to be run as a one-off job, so we define the command here.
CMD ["python", "train_model.py"]