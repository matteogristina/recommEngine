# Use a lightweight Python base image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install the dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the event generator script into the container.
COPY event_generator.py ./

# This is the command that will be run by Docker Compose.
CMD ["python", "event_generator.py"]