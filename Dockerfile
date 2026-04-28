# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/app/uploads

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# We need ffmpeg for video frame extraction and audio processing
# git and gcc are often needed for building python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    gcc \
    python3-dev \
    pkg-config \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir setuptools==69.5.1 && \
    pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Create the uploads directory
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Expose the port the app runs on
EXPOSE 8000

# Start the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
