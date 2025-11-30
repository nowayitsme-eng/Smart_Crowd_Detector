"""
YOLOv8 Crowd Detection Module
Implements REQ-1, REQ-2, REQ-3: Person detection with bounding boxes and counting
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class CrowdDetector:
    """
    Main crowd detection class using YOLOv8
    
    Satisfies SRS Requirements:
    - REQ-1: Detect individuals in video frames using pre-trained model
    - REQ-2: Display bounding boxes around detected individuals
    - REQ-3: Update count continuously as frames are processed
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the YOLOv8 detector
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.model_name = config['model']['name']
        self.confidence_threshold = config['model']['confidence_threshold']
        self.iou_threshold = config['model']['iou_threshold']
        self.device = config['model']['device']
        self.class_filter = config['model']['class_filter']
        self.min_size = config['crowd']['min_detection_size']
        
        # Performance tracking
        self.frame_times = []
        self.detection_count = 0
        
        logger.info(f"Initializing YOLOv8 Detector with model: {self.model_name}")
        logger.info(f"Device: {self.device}, Confidence: {self.confidence_threshold}")
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray, resize_factor: float = 0.5) -> Tuple[List[Dict], int, float]:
        """
        Detect people in a frame
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            resize_factor: Factor to resize frame for detection (smaller = faster)
        
        Returns:
            detections: List of detection dictionaries with bbox, confidence, class
            count: Number of people detected (REQ-3)
            processing_time: Time taken for detection
        
        Implements:
            - REQ-1: Detect individuals using YOLOv8
            - REQ-3: Count detected people
        """
        start_time = time.time()
        detections = []
        
        try:
            # Resize frame for faster detection
            if resize_factor < 1.0:
                h, w = frame.shape[:2]
                detection_frame = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)), 
                                            interpolation=cv2.INTER_NEAREST)  # Fastest interpolation
                scale_x = w / detection_frame.shape[1]
                scale_y = h / detection_frame.shape[0]
            else:
                detection_frame = frame
                scale_x = scale_y = 1.0
            
            # Run YOLOv8 inference with balanced size
            results = self.model(
                detection_frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.class_filter,  # Only detect persons (class 0)
                verbose=False,
                imgsz=256,  # Balanced size for good quality and speed
                device='cpu',
                half=False  # Use FP32 for better accuracy
            )
            
            # Extract detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Scale boxes back to original frame size
                    if resize_factor < 1.0:
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                    
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by minimum size
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width >= self.min_size and height >= self.min_size:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': 'person',
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        }
                        detections.append(detection)
            
            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)
            self.detection_count += len(detections)
            
            # Check performance constraint (SRS: ≤ 1s per frame on CPU)
            if processing_time > self.config['constraints']['max_frame_delay']:
                logger.warning(
                    f"Frame processing exceeded constraint: "
                    f"{processing_time:.3f}s > {self.config['constraints']['max_frame_delay']}s"
                )
            
            count = len(detections)
            
            return detections, count, processing_time
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [], 0, 0.0
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to display confidence scores
        
        Returns:
            frame: Frame with drawn bounding boxes (REQ-2)
        
        Implements:
            - REQ-2: Display bounding boxes around detected individuals
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            if show_confidence:
                label = f"{confidence:.0%}"  # Shorter format
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                # Background for text
                cv2.rectangle(
                    frame_copy,
                    (x1, y1 - label_size[1] - 4),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    frame_copy,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,  # Smaller font
                    (0, 0, 0),
                    1
                )
        
        return frame_copy
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics
        
        Returns:
            stats: Dictionary with performance metrics
        """
        if not self.frame_times:
            return {
                'avg_processing_time': 0.0,
                'fps': 0.0,
                'total_detections': 0,
                'frames_processed': 0
            }
        
        avg_time = np.mean(self.frame_times[-100:])  # Last 100 frames
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'avg_processing_time': avg_time,
            'fps': fps,
            'total_detections': self.detection_count,
            'frames_processed': len(self.frame_times),
            'max_processing_time': max(self.frame_times) if self.frame_times else 0.0,
            'min_processing_time': min(self.frame_times) if self.frame_times else 0.0
        }
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.frame_times = []
        self.detection_count = 0
        logger.info("Detection statistics reset")
