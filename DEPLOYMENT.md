# Zaytrics - Deployment Guide

## ⚠️ Important: Vercel Limitations

Vercel **does not support** this application because:
- Real-time video processing requires persistent connections
- OpenCV, PyTorch, and YOLOv8 are too large for serverless (500MB+ limit)
- Camera access is not available in serverless environments
- WebSocket/streaming connections are limited

## ✅ Recommended Deployment Options

### 1. **Docker + Cloud VM (Best Option)**

Deploy on AWS EC2, Google Cloud, DigitalOcean, or Azure VM:

```bash
# Create Dockerfile
docker build -t zaytrics .
docker run -p 5000:5000 --device=/dev/video0 zaytrics
```

**Providers:**
- AWS EC2 (t2.medium or better)
- Google Cloud Compute Engine
- DigitalOcean Droplet ($12/month)
- Azure Virtual Machine

### 2. **Heroku with Buildpacks**

```bash
heroku create zaytrics-crowd-monitor
heroku buildpacks:add --index 1 heroku-community/apt
heroku buildpacks:add --index 2 heroku/python
git push heroku main
```

Note: Requires camera workaround (use IP camera or video files)

### 3. **Railway.app (Easiest)**

1. Go to https://railway.app
2. Connect GitHub repository
3. Deploy automatically
4. Railway supports Docker and persistent connections

### 4. **Render.com**

1. Go to https://render.com
2. Create new Web Service
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`

### 5. **Local Network Deployment**

Run on local machine and expose via ngrok:

```bash
# Start Flask
python app.py

# In another terminal
ngrok http 5000
```

## 🐳 Docker Deployment (Recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

Deploy to any cloud provider that supports Docker.

## 📱 Alternative: Static Demo Version

If you need Vercel, I can create a lightweight demo with:
- Pre-recorded video analysis
- Static dashboard
- API endpoints only (no real-time streaming)

Would you like me to create this version?

## 🚀 Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Access at http://localhost:5000
```

## 💡 Production Tips

1. Use GPU instance for better performance
2. Set up reverse proxy (nginx) for production
3. Use PM2 or supervisor for process management
4. Enable HTTPS with Let's Encrypt
5. Set up monitoring (Sentry, DataDog)
