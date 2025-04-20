FROM ubuntu:latest
LABEL authors="romia"

ENTRYPOINT ["top", "-b"]

# Base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy all project files into container
COPY . .

# Install required packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: Create outputs folders
RUN mkdir -p outputs/saved_models outputs/results

# Run training on container start
CMD ["python", "run.py"]
