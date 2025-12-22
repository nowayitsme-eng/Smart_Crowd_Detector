"""
Zaytrics Smart Crowd Monitoring System - Web Server
Optimized for small object detection and better performance
"""

print("[*] Starting Zaytrics...")

# GPU Verification - Check CUDA availability
print("[*] Checking GPU...")
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK] GPU Detected: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    print(f"     CUDA Version: {torch.version.cuda}")
    # Set CUDA optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmul
else:
    print("[WARN] WARNING: CUDA not available, using CPU (slower)")

print("[*] Loading Flask...")
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
print("[OK] Flask loaded")

print("[*] Loading OpenCV...")
import cv2
import numpy as np
print("[OK] OpenCV loaded")

import os
import json
import time
import logging
from datetime import datetime
from threading import Thread, Lock
from queue import Queue
from collections import deque

print("[*] Loading detection modules...")
from src.detection.detector import CrowdDetector
print("[OK] Detector loaded")

from src.heatmap.generator import HeatmapGenerator
print("[OK] Heatmap loaded")

from src.video.handler import VideoHandler
print("[OK] Video handler loaded")

from src.utils.config import load_config
from src.utils.logger import setup_logger
print("[OK] All modules loaded")

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load configuration
config = load_config('config.yaml')
logger = setup_logger(config)

# Initialize components with optimized parameters for small objects
detector = CrowdDetector(config)
heatmap_generator = HeatmapGenerator(config)
video_handler = VideoHandler(config)

# Thread-safe state management
state_lock = Lock()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Check for existing video files and set default source
def get_latest_video():
    """Get the most recent uploaded video file"""
    try:
        videos_dir = 'videos'
        if os.path.exists(videos_dir):
            videos = [f for f in os.listdir(videos_dir) if allowed_file(f)]
            if videos:
                videos.sort(reverse=True)  # Sort by timestamp (filename starts with timestamp)
                return videos[0]
    except Exception as e:
        print(f"Error getting latest video: {e}")
    return None

latest_video = get_latest_video()
default_source = 'video' if latest_video else 'camera'

state = {
    'running': False,
    'heatmap_enabled': False,
    'total_detections': 0,
    'count_history': [],
    'time_history': [],
    'current_count': 0,
    'fps': 0,
    'alert_level': 'normal',
    'statistics': {},
    'last_detection_time': 0,
    'detection_cache': [],
    'frame_cache': None,
    'source_type': default_source,  # 'camera' or 'video'
    'video_file': latest_video,
    'video_loop': True  # Loop videos by default
}

print(f"Default source: {default_source}, Video file: {latest_video}")

# Use deque for frame times to prevent memory leak
frame_times = deque(maxlen=100)

# Enhanced optimization settings for GPU-accelerated crowd detection
DETECTION_INTERVAL = 4  # Run detection every 4th frame (yolov8m needs more skip)
MIN_CONFIDENCE = 0.20  # Higher for speed
RESIZE_FACTOR = 1.0   # Full resolution always
MIN_OBJECT_SIZE = 10   # Slightly larger for speed
ENABLE_MULTI_SCALE = False  # Disabled for speed

# Alert thresholds from config
WARNING_THRESHOLD = config.get('crowd', {}).get('density_threshold', 15)
CRITICAL_THRESHOLD = config.get('crowd', {}).get('warning_threshold', 25)


def update_state(key, value):
    """Thread-safe state update"""
    with state_lock:
        state[key] = value


def get_alert_level(count):
    """Determine alert level based on count (REQ-7)"""
    if count >= config['crowd']['warning_threshold']:
        return 'critical'
    elif count >= config['crowd']['density_threshold']:
        return 'warning'
    else:
        return 'normal'


def generate_frames():
    """Generate video frames with detections - supports both camera and video file"""
    global state
    
    logger.info("generate_frames() called")
    
    # Wait for running state to be true
    max_wait = 50  # 5 seconds max
    wait_count = 0
    while not state.get('running', False) and wait_count < max_wait:
        time.sleep(0.1)
        wait_count += 1
    
    if not state.get('running', False):
        logger.error("Monitoring not started, exiting generate_frames")
        return
    
    # Determine video source based on state
    with state_lock:
        source_type = state['source_type']
        video_file = state['video_file']
    
    logger.info(f"Source type: {source_type}, Video file: {video_file}")
    logger.info(f"Will use: {'VIDEO FILE' if (source_type == 'video' and video_file) else 'CAMERA'}")
    
    if source_type == 'video' and video_file:
        logger.info(f"Opening video file: {video_file}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file)
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            # Generate error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Video File Not Found", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        # Set video source properly
        video_handler.set_source(video_path, is_camera=False)
        logger.info(f"Set video source to: {video_path}")
    else:
        logger.info("Opening camera source")
        # Set camera source properly
        video_handler.set_source(0, is_camera=True)
        logger.info("Set camera source to: 0")
    
    # Try to open video source with retry logic
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        if video_handler.open():
            break
        retry_count += 1
        logger.warning(f"Failed to open video source, retry {retry_count}/{max_retries}")
        time.sleep(1)
    
    if retry_count >= max_retries:
        logger.error("Failed to open video source after retries")
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Not Available", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    logger.info("Video source opened successfully")
    frame_count = 0
    start_time = time.time()
    
    # Caching for frame skipping
    last_detections = []
    last_count = 0
    last_annotated_frame = None  # Initialize to prevent NameError
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    try:
        while state['running']:
            ret, frame = video_handler.read_frame()
            
            if not ret:
                # Handle video loop on read failure
                if state['source_type'] == 'video' and state['video_loop']:
                    logger.info("Video ended, restarting loop...")
                    if video_handler.restart():
                        frame_count = 0
                        start_time = time.time()
                        consecutive_failures = 0
                        logger.info("Video loop restarted successfully")
                        continue
                
                # For non-looping videos or cameras, count failures
                consecutive_failures += 1
                logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive frame read failures")
                    break
                
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0  # Reset on successful read
            
            # Apply resize factor if configured (performance optimization)
            if RESIZE_FACTOR < 1.0:
                new_width = int(frame.shape[1] * RESIZE_FACTOR)
                new_height = int(frame.shape[0] * RESIZE_FACTOR)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            frame_count += 1
            
            # Run detection based on configured interval (GPU-optimized)
            # Run on frames 0, DETECTION_INTERVAL, DETECTION_INTERVAL*2, etc.
            should_detect = (frame_count - 1) % DETECTION_INTERVAL == 0
            if should_detect:
                detections, count, detection_time = detector.detect(frame)
                last_detections = detections
                last_count = count
                
                # Choose display mode: heatmap-only OR bounding boxes
                if state['heatmap_enabled']:
                    # Heatmap mode: Skip bounding boxes for cleaner visualization
                    frame_display, heatmap_time = heatmap_generator.generate_heatmap(
                        frame, detections  # Removed unnecessary copy - generator copies internally
                    )
                else:
                    # Normal mode: Draw bounding boxes
                    frame_display = detector.draw_detections(frame.copy(), detections)
                
                # Cache the annotated frame for reuse (OPTIMIZED)
                last_annotated_frame = frame_display.copy()
            else:
                # Reuse cached annotated frame instead of re-drawing (MAJOR OPTIMIZATION)
                detections = last_detections
                count = last_count
                if last_annotated_frame is not None:
                    frame_display = last_annotated_frame
                else:
                    frame_display = detector.draw_detections(frame, detections)
            
            # Update state with proper locking to prevent race conditions
            with state_lock:
                state['current_count'] = count
                # Only track current frame count, not accumulating total (prevents infinite growth)
                state['last_detection_time'] = time.time()
                
                # Update alert level based on configurable thresholds
                if count >= CRITICAL_THRESHOLD:
                    state['alert_level'] = 'critical'
                elif count >= WARNING_THRESHOLD:
                    state['alert_level'] = 'warning'
                else:
                    state['alert_level'] = 'normal'
            
            # Debug log for detection count (reduced logging frequency)
            if count > 0 and frame_count % 30 == 0:  # Log every 30 frames instead of every frame
                logger.debug(f"Detected {count} people in frame {frame_count}")
            
            # Calculate FPS using deque for memory efficiency
            current_time = time.time()
            frame_times.append(current_time)
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                # Update FPS with state lock
                with state_lock:
                    state['fps'] = len(frame_times) / elapsed if elapsed > 0 else 0
            
            # Encode frame to JPEG with good quality (70% - balance quality and bandwidth)
            ret, buffer = cv2.imencode('.jpg', frame_display, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    except Exception as e:
        logger.error(f"Error in generate_frames: {e}", exc_info=True)
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Processing Error", (180, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_frame, "Check logs for details", (150, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        video_handler.release()
        logger.info("Video handler released")
        # Clear frame times on exit
        frame_times.clear()


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route with optimized buffering"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    })


@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start monitoring (REQ-6)"""
    update_state('running', True)
    logger.info("Monitoring started")
    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    update_state('running', False)
    logger.info("Monitoring stopped")
    return jsonify({'status': 'stopped'})


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload a video file for processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Validate video file can be opened
        test_cap = None
        try:
            test_cap = cv2.VideoCapture(filepath)
            if not test_cap.isOpened():
                os.remove(filepath)  # Delete invalid file
                return jsonify({'error': 'Invalid video file. Cannot be opened by OpenCV.'}), 400
            
            # Verify it has frames
            ret, test_frame = test_cap.read()
            if not ret or test_frame is None:
                os.remove(filepath)
                return jsonify({'error': 'Invalid video file. No readable frames.'}), 400
        finally:
            if test_cap is not None:
                test_cap.release()
        
        # Update state to use video file
        with state_lock:
            state['source_type'] = 'video'
            state['video_file'] = filename
            state['video_loop'] = request.form.get('loop', 'false').lower() == 'true'
        
        logger.info(f"Video uploaded successfully: {filename}")
        return jsonify({
            'status': 'success',
            'filename': filename,
            'source_type': 'video'
        })
    except Exception as e:
        logger.error(f"Error uploading video: {e}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/switch_source', methods=['POST'])
def switch_source():
    """Switch between camera and video file"""
    data = request.get_json()
    source_type = data.get('source_type', 'camera')
    
    # Stop current monitoring if running
    with state_lock:
        was_running = state['running']
        state['running'] = False
    
    time.sleep(0.5)  # Allow current stream to stop
    
    # Update source - ENSURE camera mode clears video file
    with state_lock:
        state['source_type'] = source_type
        if source_type == 'camera':
            state['video_file'] = None
            logger.info("Camera mode activated - cleared video file from state")
        else:
            logger.info(f"Video mode - current video: {state.get('video_file', 'None')}")
    
    logger.info(f"Switched to {source_type} source")
    
    return jsonify({
        'status': 'success',
        'source_type': source_type,
        'was_running': was_running
    })


@app.route('/api/list_videos', methods=['GET'])
def list_videos():
    """List available uploaded videos"""
    try:
        videos = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                videos.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
        return jsonify({'videos': videos})
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/toggle_heatmap', methods=['POST'])
def toggle_heatmap():
    """Toggle heatmap (REQ-8, REQ-9)"""
    with state_lock:
        state['heatmap_enabled'] = not state['heatmap_enabled']
    logger.info(f"Heatmap {'enabled' if state['heatmap_enabled'] else 'disabled'}")
    return jsonify({'heatmap_enabled': state['heatmap_enabled']})


@app.route('/api/reset', methods=['POST'])
def reset_statistics():
    """Reset statistics"""
    with state_lock:
        state['total_detections'] = 0
        state['count_history'] = []
        state['time_history'] = []
    logger.info("Statistics reset")
    return jsonify({'status': 'reset'})


@app.route('/api/optimize', methods=['POST'])
def optimize_detection():
    """Manual optimization endpoint for small objects"""
    global MIN_CONFIDENCE, DETECTION_INTERVAL, RESIZE_FACTOR, ENABLE_MULTI_SCALE
    
    data = request.get_json()
    if data:
        MIN_CONFIDENCE = data.get('confidence', MIN_CONFIDENCE)
        DETECTION_INTERVAL = max(1, data.get('interval', DETECTION_INTERVAL))
        RESIZE_FACTOR = min(1.0, max(0.3, data.get('resize_factor', RESIZE_FACTOR)))
        ENABLE_MULTI_SCALE = data.get('multi_scale', ENABLE_MULTI_SCALE)
    
    logger.info(f"Small object optimization applied: confidence={MIN_CONFIDENCE}, interval={DETECTION_INTERVAL}")
    return jsonify({
        'confidence': MIN_CONFIDENCE,
        'interval': DETECTION_INTERVAL,
        'resize_factor': RESIZE_FACTOR,
        'multi_scale': ENABLE_MULTI_SCALE,
        'min_object_size': MIN_OBJECT_SIZE
    })


@app.route('/api/stats')
def get_statistics():
    """Get current statistics (REQ-6, REQ-7)"""
    with state_lock:
        return jsonify({
            'count': state['current_count'],
            'fps': round(state['fps'], 1),
            'alert_level': state['alert_level'],
            'total_detections': state['total_detections'],
            'running': state['running'],
            'heatmap_enabled': state['heatmap_enabled'],
            'count_history': state['count_history'][-50:],
            'time_history': state['time_history'][-50:],
            'thresholds': {
                'warning': config['crowd']['density_threshold'],
                'critical': config['crowd']['warning_threshold']
            },
            'optimization': {
                'confidence': MIN_CONFIDENCE,
                'detection_interval': DETECTION_INTERVAL,
                'resize_factor': RESIZE_FACTOR,
                'multi_scale': ENABLE_MULTI_SCALE,
                'min_object_size': MIN_OBJECT_SIZE
            }
        })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify({
        'video_source': config['video']['source'],
        'confidence_threshold': config['model']['confidence_threshold'],
        'density_threshold': config['crowd']['density_threshold'],
        'warning_threshold': config['crowd']['warning_threshold'],
        'small_object_optimization': {
            'min_confidence': MIN_CONFIDENCE,
            'detection_interval': DETECTION_INTERVAL,
            'resize_factor': RESIZE_FACTOR,
            'multi_scale': ENABLE_MULTI_SCALE,
            'min_object_size': MIN_OBJECT_SIZE
        }
    })


@app.route('/api/health')
def health_check():
    """System health check"""
    with state_lock:
        return jsonify({
            'status': 'healthy',
            'running': state['running'],
            'fps': state['fps'],
            'current_count': state['current_count'],
            'timestamp': datetime.now().isoformat()
        })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting Enhanced Zaytrics Web Server (Small Object Optimized)")
    logger.info(f"Access the dashboard at: http://localhost:{port}")
    logger.info("Small Object Detection Optimizations:")
    logger.info(f"  - Detection interval: {DETECTION_INTERVAL} frames")
    logger.info(f"  - Minimum confidence: {MIN_CONFIDENCE}")
    logger.info(f"  - Resize factor: {RESIZE_FACTOR}")
    logger.info(f"  - Multi-scale detection: {ENABLE_MULTI_SCALE}")
    logger.info(f"  - Minimum object size: {MIN_OBJECT_SIZE} pixels")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)