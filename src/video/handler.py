"""
Video Input Handler
Manages video input from camera or file
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
import os

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
        
        logger.info(f"Video Handler initialized with source: {self.source}")
    
    def open(self) -> bool:
        """
        Open video source
        
        Returns:
            success: True if video source opened successfully
        """
        try:
            # Determine if source is camera or file
            if isinstance(self.source, int):
                self.is_camera = True
                logger.info(f"Opening webcam: Camera {self.source}")
            else:
                self.is_camera = False
                if not os.path.exists(self.source):
                    logger.error(f"Video file not found: {self.source}")
                    return False
                logger.info(f"Opening video file: {self.source}")
            
            # Open video source with DirectShow backend for Windows cameras
            if self.is_camera:
                # Try with DirectShow backend first (Windows)
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    logger.warning("DirectShow backend failed, trying default backend")
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error("Failed to open video source")
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from video source
        
        Returns:
            success: True if frame read successfully
            frame: Frame as numpy array (BGR format)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        try:
            # Skip frames if configured (using grab for speed)
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
        """Release video source"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video source released")
    
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
