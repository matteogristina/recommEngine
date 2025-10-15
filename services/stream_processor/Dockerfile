# Use a lightweight Python base image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install the dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the stream processor script into the container.
COPY stream_processor.py ./

# This is the command to run the script. It is a long-running process.
CMD ["python", "stream_processor.py"]