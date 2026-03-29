"""
YOLOv8 Crowd Detection Module - Enhanced for Small Objects
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
    Main crowd detection class using YOLOv8 - Enhanced for small objects
    
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
        
        # Optimization parameters
        self.small_object_mode = config['model'].get('small_object_mode', True)
        self.imgsz = 416  # Lower resolution for faster TensorRT inference
        
        # Dynamic mode parameters (can be updated via API)
        self.max_det = 300  # Default max detections
        self.second_pass_conf = 0.05  # Default second pass confidence
        self.duplicate_threshold = 30  # Default duplicate detection threshold
        self.min_box_size = 5  # Default minimum box size
        
        # Performance tracking
        from collections import deque
        self.frame_times = deque(maxlen=30)  # Keep last 30 frame times
        self.detection_count = 0
        self.frame_count = 0  # Track frames for logging throttle
        
        logger.info(f"Initializing YOLOv8 Detector with model: {self.model_name}")
        logger.info(f"Device: {self.device}, Confidence: {self.confidence_threshold}")
        logger.info(f"Small object mode: {self.small_object_mode}")
        
        # Check for TensorRT optimized model first
        tensorrt_model = self.model_name.replace('.pt', '.engine')
        use_tensorrt = False
        
        try:
            import os
            if os.path.exists(tensorrt_model):
                logger.info(f"Loading TensorRT optimized model: {tensorrt_model}")
                self.model = YOLO(tensorrt_model)
                use_tensorrt = True
            else:
                logger.info(f"Loading PyTorch model: {self.model_name}")
                logger.info(f"TIP: Export to TensorRT for 2-3x speedup: yolo export model={self.model_name} format=engine half=True device=0")
                self.model = YOLO(self.model_name)
                self.model.to(self.device)
                use_tensorrt = False
            
            logger.info("YOLOv8 model loaded successfully")
            logger.info(f"*** USING {'TensorRT ENGINE' if use_tensorrt else 'PyTorch FP16'} for inference ***")
            
            # GPU Warmup - run dummy inference to compile CUDA kernels
            logger.info("Warming up GPU (this may take a few seconds)...")
            dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
            for _ in range(5):  # Run 5 warmup passes for better optimization
                self.model(dummy_frame, conf=0.5, verbose=False, device=0, half=True, imgsz=self.imgsz)
            logger.info("GPU warmup complete - ready for fast inference")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Light preprocessing - only applied if needed
        Returns frame as-is for speed (YOLO handles normalization)
        """
        return frame  # Skip preprocessing for speed

    def detect_additional_pass(self, frame: np.ndarray, existing_detections: List[Dict]) -> List[Dict]:
        """
        Additional detection pass with very low confidence for missed small objects
        """
        try:
            # Second pass with lower confidence threshold for small/distant objects
            results = self.model(
                frame,
                conf=self.second_pass_conf,  # Use dynamic second pass confidence
                iou=self.iou_threshold,
                classes=self.class_filter,
                verbose=False,
                imgsz=self.imgsz,
                device=self.device,
                half=True
            )
            
            additional_detections = []
            existing_centers = [(d['center'][0], d['center'][1]) for d in existing_detections]
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    width = x2 - x1
                    height = y2 - y1
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Filter out very small noise using dynamic min_box_size
                    if width < self.min_box_size or height < self.min_box_size:
                        continue
                    
                    # Check if this is a duplicate (near existing detection)
                    is_duplicate = False
                    for ex, ey in existing_centers:
                        distance = ((center_x - ex)**2 + (center_y - ey)**2)**0.5
                        if distance < self.duplicate_threshold:  # Use dynamic threshold
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': 'person',
                            'center': [center_x, center_y],
                            'size': 'tiny' if (width < 10 or height < 10) else ('small' if (width < 50 or height < 50) else 'normal')
                        }
                        additional_detections.append(detection)
            
            return additional_detections
            
        except Exception as e:
            logger.error(f"Additional detection pass error: {e}")
            return []
    
    def detect(self, frame: np.ndarray, resize_factor: float = 1.0, 
               confidence_threshold: float = None) -> Tuple[List[Dict], int, float]:
        """
        Detect people in the frame using YOLOv8 with TensorRT
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            resize_factor: Ignored - always uses full resolution for best accuracy
            confidence_threshold: Optional override for detection threshold
        
        Returns:
            detections: List of all detected people (primary + second pass)
            count: Total number of people detected
            processing_time: Time taken for detection
        """
        start_time = time.time()
        detections = []
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        try:
            # Primary detection with CUDA using configured device
            results = self.model(
                frame,
                conf=confidence_threshold,
                iou=self.iou_threshold,
                classes=self.class_filter,
                verbose=False,
                imgsz=self.imgsz,
                device=self.device,
                half=True,  # FP16 inference for 2x speedup on RTX 3050
                max_det=self.max_det,  # Use dynamic max detections
                agnostic_nms=False,
                retina_masks=False  # Disable for speed
            )
            
            # Extract primary detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Filter by minimum size to remove noise
                    if width >= self.min_size and height >= self.min_size:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': 'person',
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'size': 'tiny' if (width < 10 or height < 10) else ('small' if (width < 50 or height < 50) else 'normal')
                        }
                        detections.append(detection)
            
            # Second pass for better small object detection
            if self.small_object_mode:
                additional = self.detect_additional_pass(frame, detections)
                detections.extend(additional)
            
            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)
            self.detection_count += len(detections)
            self.frame_count += 1
            
            count = len(detections)
            
            # Log detection count occasionally (every 30 frames) to avoid log spam
            if self.frame_count % 30 == 0 or count > 0:
                logger.debug(f"Detected {count} people in {processing_time:.3f}s (frame {self.frame_count})")
            
            return detections, count, processing_time
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0, 0.0

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes with edge case handling
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Calculate intersection area
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Handle edge cases
        if box1_area <= 0 or box2_area <= 0:
            return 0.0
        
        union_area = box1_area + box2_area - inter_area
        
        # Avoid division by zero
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes with high visibility for crowd detection
        Color-coded by confidence level
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to display confidence scores
        
        Returns:
            frame: Frame with drawn bounding boxes
        """
        frame_copy = frame.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            is_small = det.get('size') == 'small'
            
            # Color by confidence: Green (high) -> Yellow (medium) -> Orange (low)
            if confidence >= 0.5:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence >= 0.25:
                color = (0, 255, 255)  # Yellow - medium
            elif confidence >= 0.15:
                color = (0, 165, 255)  # Orange - lower
            else:
                color = (0, 128, 255)  # Light orange - very low
            
            thickness = 1 if is_small else 2
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center dot for all detections
            center_x, center_y = det['center']
            cv2.circle(frame_copy, (center_x, center_y), 2, color, -1)
        
        # Draw prominent count display in top-left corner
        count = len(detections)
        count_text = f"PEOPLE: {count}"
        
        # Background box for count
        (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame_copy, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (5, 5), (text_w + 15, text_h + 15), (0, 255, 0), 2)
        
        # Count text
        cv2.putText(frame_copy, count_text, (10, text_h + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
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
        from collections import deque
        self.frame_times = deque(maxlen=30)
        self.detection_count = 0
        self.frame_count = 0
        logger.info("Detection statistics reset")