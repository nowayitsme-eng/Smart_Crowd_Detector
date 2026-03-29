FROM python:3.11-slim

# Create user with UID 1000 - Hugging Face requires this specific setup
RUN useradd -m -u 1000 user

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy dependencies first (for Docker caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

# Hugging Face exposes exactly port 7860
ENV PORT=7860
EXPOSE 7860

# Run the Flask App
CMD ["python", "app.py"]
