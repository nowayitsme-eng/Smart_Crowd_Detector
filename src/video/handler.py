"""
Video Input Handler
Manages video input from camera or file
Optimized with threaded capture for GPU inference
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
import os
import threading
from collections import deque

logger = logging.getLogger(__name__)


class VideoHandler:
    """
    Handle video input from webcam or video file
    
    Supports:
    - Webcam input (source = 0, 1, 2, ...)
    - Video file input (source = path to file)
    - Frame skipping for performance
    - Resolution configuration
    """
    
    def __init__(self, config: dict):
        """
        Initialize video handler
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.source = config['video']['source']
        self.target_fps = config['video']['fps']
        self.skip_frames = config['video']['skip_frames']
        self.target_width = config['video']['resolution']['width']
        self.target_height = config['video']['resolution']['height']
        
        self.cap = None
        self.frame_count = 0
        self.is_camera = False
        
        # Threaded capture for async frame reading (cameras only)
        self._thread = None
        self._stopped = False
        self._frame_buffer = deque(maxlen=2)  # Small buffer for low latency
        self._lock = threading.Lock()
        self._use_threading = False  # Disabled - causes issues with video files
        
        logger.info(f"Video Handler initialized with source: {self.source}")
    
    def set_source(self, source, is_camera: bool = None):
        """
        Set video source and optionally specify if it's a camera
        
        Args:
            source: Video source (int for camera, str for file path)
            is_camera: Explicitly specify if source is camera (optional)
        """
        self.source = source
        
        if is_camera is not None:
            self.is_camera = is_camera
            self._is_camera_explicit = True  # Mark as explicitly set
        else:
            # Auto-detect
            self.is_camera = isinstance(source, int)
            if hasattr(self, '_is_camera_explicit'):
                delattr(self, '_is_camera_explicit')
        
        logger.info(f"Source set to: {source}, is_camera: {self.is_camera}")
    
    def open(self) -> bool:
        """
        Open video source
        
        Returns:
            success: True if video source opened successfully
        """
        try:
            # Release any existing capture
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Validate source type
            if isinstance(self.source, int):
                # Camera source - only update is_camera if not already explicitly set
                if not hasattr(self, '_is_camera_explicit'):
                    self.is_camera = True
                logger.info(f"Opening webcam: Camera {self.source}")
            elif isinstance(self.source, str):
                # File source - only update is_camera if not already explicitly set
                if not hasattr(self, '_is_camera_explicit'):
                    self.is_camera = False
                if not os.path.exists(self.source):
                    logger.error(f"Video file not found: {self.source}")
                    return False
                logger.info(f"Opening video file: {self.source}")
            else:
                logger.error(f"Invalid source type: {type(self.source)}")
                return False
            
            # Open video source with DirectShow backend for Windows cameras
            if self.is_camera:
                # Try with DirectShow backend first (Windows)
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    logger.warning("DirectShow backend failed, trying default backend")
                    self.cap = cv2.VideoCapture(self.source)
            else:
                # For video files, use default backend
                logger.info(f"Creating VideoCapture for file: {self.source}")
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set camera properties if using webcam
            if self.is_camera:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # MJPEG for speed
            
            # Get actual video properties
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video source opened successfully")
            logger.info(f"  Resolution: {actual_width}x{actual_height}")
            logger.info(f"  FPS: {actual_fps}")
            if not self.is_camera:
                logger.info(f"  Total frames: {total_frames}")
            
            # Start threaded capture for async frame reading
            if self._use_threading:
                self._start_capture_thread()
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False
    
    def _start_capture_thread(self):
        """Start background thread for frame capture"""
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Threaded frame capture started")
    
    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        while not self._stopped and self.cap is not None and self.cap.isOpened():
            # Skip frames if configured
            for _ in range(self.skip_frames - 1):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            
            if ret:
                # Resize if needed
                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height),
                                       interpolation=cv2.INTER_NEAREST)
                
                with self._lock:
                    self._frame_buffer.append((True, frame))
            else:
                with self._lock:
                    self._frame_buffer.append((False, None))
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from video source
        Uses threaded capture if enabled for non-blocking reads
        
        Returns:
            success: True if frame read successfully
            frame: Frame as numpy array (BGR format)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        try:
            # Use threaded capture if enabled
            if self._use_threading and self._thread is not None:
                with self._lock:
                    if len(self._frame_buffer) > 0:
                        ret, frame = self._frame_buffer.pop()
                        if ret:
                            self.frame_count += 1
                        return ret, frame
                    else:
                        return False, None
            
            # Fallback to synchronous capture
            for _ in range(self.skip_frames - 1):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            
            if not ret:
                return False, None
            
            # Only resize if dimensions don't match (optimization)
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height), 
                                  interpolation=cv2.INTER_NEAREST)  # Fastest interpolation
            
            self.frame_count += 1
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def release(self):
        """Release video source and stop capture thread with proper cleanup"""
        try:
            self._stopped = True
            
            # Wait for thread to finish with extended timeout
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=3.0)  # Increased from 1.0 to 3.0 seconds
                
                # Force cleanup if thread is still alive
                if self._thread.is_alive():
                    logger.warning("Capture thread did not stop within timeout, forcing cleanup")
                
                self._thread = None
            
            # Release OpenCV capture
            if self.cap is not None:
                if self.cap.isOpened():
                    self.cap.release()
                self.cap = None
                logger.info("Video source released")
            
            # Clear buffer
            with self._lock:
                self._frame_buffer.clear()
                
        except Exception as e:
            logger.error(f"Error releasing video capture: {e}")
            # Force cleanup even on error
            self.cap = None
            self._thread = None
    
    def is_opened(self) -> bool:
        """Check if video source is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_properties(self) -> dict:
        """
        Get video source properties
        
        Returns:
            properties: Dictionary with video properties
        """
        if self.cap is None or not self.cap.isOpened():
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'current_frame': self.frame_count,
            'is_camera': self.is_camera
        }
    
    def is_video_end(self) -> bool:
        """
        Check if video file has reached the end
        
        Returns:
            True if video has ended, False otherwise (or if camera)
        """
        if self.is_camera or self.cap is None:
            return False
        
        try:
            # Get current position and total frames
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            # Check if we're at or past the end
            # Use -2 threshold to catch end before actual failure
            if total_frames > 0 and current_pos >= total_frames - 2:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking video end: {e}")
            return False
    
    def restart(self) -> bool:
        """
        Restart video from beginning (for video files only)
        
        Returns:
            success: True if restart successful
        """
        if self.is_camera:
            logger.warning("Cannot restart camera feed")
            return False
        
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            logger.info("Video restarted from beginning")
            return True
        
        return False
    
    def __del__(self):
        """Destructor to ensure video source is released"""
        self.release()
