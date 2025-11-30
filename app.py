"""
Zaytrics Smart Crowd Monitoring System - Web Server
Simple Flask server with HTML/CSS/JS frontend
"""

print("🚀 Starting Zaytrics...")
print("📦 Loading Flask...")
from flask import Flask, render_template, Response, jsonify
print("✅ Flask loaded")

print("📦 Loading OpenCV...")
import cv2
import numpy as np
print("✅ OpenCV loaded")

import json
import time
import logging
from datetime import datetime
from threading import Thread
from queue import Queue

print("📦 Loading detection modules...")
from src.detection.detector import CrowdDetector
print("✅ Detector loaded")

from src.heatmap.generator import HeatmapGenerator
print("✅ Heatmap loaded")

from src.video.handler import VideoHandler
print("✅ Video handler loaded")

from src.utils.config import load_config
from src.utils.logger import setup_logger
print("✅ All modules loaded")

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Load configuration
config = load_config('config.yaml')
logger = setup_logger(config)

# Initialize components
detector = CrowdDetector(config)
heatmap_generator = HeatmapGenerator(config)
video_handler = VideoHandler(config)

# Global state
state = {
    'running': False,
    'heatmap_enabled': False,
    'total_detections': 0,
    'count_history': [],
    'time_history': [],
    'current_count': 0,
    'fps': 0,
    'alert_level': 'normal',
    'statistics': {}
}


def get_alert_level(count):
    """Determine alert level based on count (REQ-7)"""
    if count >= config['crowd']['warning_threshold']:
        return 'critical'
    elif count >= config['crowd']['density_threshold']:
        return 'warning'
    else:
        return 'normal'


def generate_frames():
    """Generate video frames with detections"""
    global state
    
    logger.info("Attempting to open video source...")
    if not video_handler.open():
        logger.error("Failed to open video source")
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Not Available", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    logger.info("Video source opened successfully")
    frame_count = 0
    start_time = time.time()
    
    try:
        while state['running']:
            ret, frame = video_handler.read_frame()
            
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            frame_count += 1
            
            # Detect people (REQ-1) - balanced quality/speed
            detections, count, detection_time = detector.detect(frame, resize_factor=0.5)
            
            # Draw bounding boxes (REQ-2)
            frame_display = detector.draw_detections(frame, detections)
            
            # Generate heatmap if enabled (REQ-4, REQ-5)
            if state['heatmap_enabled']:
                frame_display, heatmap_time = heatmap_generator.generate_heatmap(
                    frame_display, detections
                )
            
            # Update state
            state['current_count'] = count
            state['total_detections'] += count
            state['alert_level'] = get_alert_level(count)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                state['fps'] = frame_count / elapsed
            
            # Store history (last 100 points)
            current_time = datetime.now().strftime('%H:%M:%S')
            state['time_history'].append(current_time)
            state['count_history'].append(count)
            if len(state['count_history']) > 100:
                state['time_history'].pop(0)
                state['count_history'].pop(0)
            
            # Encode frame as JPEG with optimized settings
            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY), 85,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,  # Optimize encoding
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0  # Disable progressive for speed
            ]
            ret, buffer = cv2.imencode('.jpg', frame_display, encode_param)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        logger.error(f"Error in generate_frames: {e}")
    finally:
        video_handler.release()
        logger.info("Video handler released")


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start monitoring (REQ-6)"""
    state['running'] = True
    logger.info("Monitoring started")
    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    state['running'] = False
    logger.info("Monitoring stopped")
    return jsonify({'status': 'stopped'})


@app.route('/api/toggle_heatmap', methods=['POST'])
def toggle_heatmap():
    """Toggle heatmap (REQ-8, REQ-9)"""
    state['heatmap_enabled'] = not state['heatmap_enabled']
    logger.info(f"Heatmap {'enabled' if state['heatmap_enabled'] else 'disabled'}")
    return jsonify({'heatmap_enabled': state['heatmap_enabled']})


@app.route('/api/reset', methods=['POST'])
def reset_statistics():
    """Reset statistics"""
    state['total_detections'] = 0
    state['count_history'] = []
    state['time_history'] = []
    logger.info("Statistics reset")
    return jsonify({'status': 'reset'})


@app.route('/api/stats')
def get_statistics():
    """Get current statistics (REQ-6, REQ-7)"""
    return jsonify({
        'count': state['current_count'],
        'fps': round(state['fps'], 1),
        'alert_level': state['alert_level'],
        'total_detections': state['total_detections'],
        'running': state['running'],
        'heatmap_enabled': state['heatmap_enabled'],
        'count_history': state['count_history'],
        'time_history': state['time_history'],
        'thresholds': {
            'warning': config['crowd']['density_threshold'],
            'critical': config['crowd']['warning_threshold']
        }
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify({
        'video_source': config['video']['source'],
        'confidence_threshold': config['model']['confidence_threshold'],
        'density_threshold': config['crowd']['density_threshold'],
        'warning_threshold': config['crowd']['warning_threshold']
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting Zaytrics Web Server")
    logger.info(f"Access the dashboard at: http://localhost:{port}")
    logger.info(f"Static folder: {app.static_folder}")
    logger.info(f"Template folder: {app.template_folder}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
