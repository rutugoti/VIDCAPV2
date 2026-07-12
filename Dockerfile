# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (ffmpeg and ffprobe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ /app/src/

# Define build arguments and bake them as environment variables at build time
ARG GROQ_API_KEY
ARG GROQ_API_KEYS
ENV GROQ_API_KEY=$GROQ_API_KEY
ENV GROQ_API_KEYS=$GROQ_API_KEYS

# Run the agent when the container launches
CMD ["python", "-m", "src.agent"]
