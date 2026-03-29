---
title: Smart Crowd Detector
emoji: 🏃
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Smart Crowd Detector

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An AI-powered real-time crowd detection and monitoring system using YOLOv8 with an adaptive heatmap visualization. Built for safety, efficiency, and smart crowd management.

## 🎯 Features

- **Real-time People Detection** - GPU-accelerated YOLOv8 model with TensorRT, OpenCV, and PyTorch.
- **Dynamic Detection Modes** - Switch dynamically depending on your environment:
  - 🟢 **Stadium Mode**: High density optimization (500 max detections, low confidence/IOU thresholds) perfectly tailored for massive crowds and distant individuals.
  - 👥 **Normal Mode**: Balanced accuracy and framing for standard environments (offices, retail).
  - ⚡ **Fast Mode**: Maximum framerate via skipped frames and lower resolutions for basic needs on lower-end hardware.
- **Adaptive Heatmaps** - Smart kernel sizing based on object distance to represent crowd density accurate to visual depth.
- **Dual Source Input** - Works seamlessly with an active webcam or video files with continuous looping playback support.
- **Alert System** - Highly configurable warning/critical crowd density thresholds.
- **High Performance** - Thread-safe state caching, frame deferring, GPU pipeline offloading, and optimized resolutions delivering 30-35 FPS on fast mode or extreme accuracy insights for dense stadium applications.
- **Modern Dashboard** - Clean F1-themed interface built on vanilla JS and standard web sockets.

## 📋 Requirements

- **GPU**: CUDA-capable NVIDIA GPU (RTX 3050+) is heavily recommended for Stadium mode.
- **RAM**: 8GB minimum, 16GB recommended.
- **Software**: Python 3.10+, CUDA 12.0+

## 🚀 Installation & Local Setup

```bash
# 1. Clone the Repository
git clone https://github.com/nowayitsme-eng/Smart_Crowd_Detector.git
cd Smart_Crowd_Detector

# 2. Create and Activate Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```
*Access the dashboard at `http://localhost:5000`*

## 🐳 Docker & Cloud Deployment

⚠️ **Note on Serverless (e.g. Vercel)**: Standard serverless does not support this application. Real-time video processing requires persistent connections, background sockets, and large ML sizes which bypass standard serverless constraints. Use Docker or standard scalable containers.

### Docker (Recommended for AWS, GCP, Azure, DigitalOcean)

Create a `Dockerfile` with the following:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```
Build and run:
```bash
docker build -t app-monitor .
docker run -p 5000:5000 --device=/dev/video0 app-monitor
```

### Render.com / Railway.app
- Connect your GitHub Repository natively.
- **Railway**: Instantly supported utilizing persistent Docker builds.
- **Render**: Use a standard Web Service provider. Build command: `pip install -r requirements.txt`. Start command: `python app.py`.

### Heroku
Use Heroku buildpacks if not building via container setup:
```bash
heroku create smart-crowd-monitor
heroku buildpacks:add --index 1 heroku-community/apt
heroku buildpacks:add --index 2 heroku/python
git push heroku main
```

## ⚙️ Configuration

Tweak main properties centrally in `config.yaml`:
```yaml
video:
  source: 0  # Camera index or path/to/video.mp4
  fps: 30
  resolution:
    width: 640
    height: 480

model:
  confidence_threshold: 0.35
  device: "cuda"  # switch to "cpu" if no GPU available

crowd:
  density_threshold: 20
  warning_threshold: 35
```

## 📄 License & Credits

MIT License

**Author:** Ali Abdullah - [GitHub: nowayitsme-eng](https://github.com/nowayitsme-eng)  
**Acknowledgments:** YOLOv8 by Ultralytics, OpenCV, Flask Framework


