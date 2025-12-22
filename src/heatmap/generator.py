"""
Crowd Density Heatmap Generator
Implements REQ-4, REQ-5: Generate and visualize crowd density zones
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """
    Generate crowd density heatmaps
    
    Satisfies SRS Requirements:
    - REQ-4: Generate localized density zones
    - REQ-5: Apply color map representing crowd concentration
    """
    
    def __init__(self, config: Dict):
        """
        Initialize heatmap generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config['heatmap']['enabled']
        self.kernel_size = config['heatmap']['kernel_size']
        self.alpha = config['heatmap']['alpha']
        self.colormap_name = config['heatmap']['colormap']
        
        # Adaptive heatmap settings
        self.adaptive = config['heatmap'].get('adaptive', True)
        self.min_kernel_size = config['heatmap'].get('min_kernel_size', 30)
        self.max_kernel_size = config['heatmap'].get('max_kernel_size', 150)
        self.blur_strength = config['heatmap'].get('blur_strength', 0.6)
        
        # Map colormap name to OpenCV constant
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'cool': cv2.COLORMAP_COOL
        }
        
        self.colormap = colormap_dict.get(self.colormap_name, cv2.COLORMAP_JET)
        
        # Performance tracking
        self.generation_times = []
        
        logger.info(f"Heatmap Generator initialized: Enabled={self.enabled}")
        logger.info(f"Kernel size: {self.kernel_size}, Alpha: {self.alpha}, Colormap: {self.colormap_name}")
        logger.info(f"Adaptive mode: {self.adaptive}, Range: {self.min_kernel_size}-{self.max_kernel_size}")
    
    def generate_heatmap(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, float]:
        """
        Generate crowd density heatmap with ADAPTIVE kernel sizing
        
        Args:
            frame: Input frame
            detections: List of detections with center points and bbox
        
        Returns:
            heatmap_overlay: Frame with heatmap overlay
            generation_time: Time taken to generate heatmap
        
        Implements:
            - REQ-4: Generate localized density zones
            - REQ-5: Apply color map for crowd concentration
            - ADAPTIVE: Auto-adjusts kernel size based on detection box dimensions
        """
        start_time = time.time()
        
        # Validate inputs
        if frame is None or frame.size == 0:
            logger.error("Invalid frame provided to heatmap generator")
            return np.zeros((480, 640, 3), dtype=np.uint8), 0.0
        
        # Only check if there are detections
        if not detections or len(detections) == 0:
            generation_time = time.time() - start_time
            return frame.copy(), generation_time
        
        try:
            h, w = frame.shape[:2]
            
            # Validate frame dimensions
            if h <= 0 or w <= 0:
                logger.error(f"Invalid frame dimensions: {h}x{w}")
                return frame.copy(), 0.0
            
            # Create empty density map (REQ-4: localized density zones)
            density_map = np.zeros((h, w), dtype=np.float32)
            
            # Calculate adaptive kernel size with validation
            if self.adaptive and len(detections) > 0:
                total_size = 0
                valid_detections = 0
                
                for det in detections:
                    try:
                        bbox = det.get('bbox', [])
                        if len(bbox) != 4:
                            continue
                        
                        x1, y1, x2, y2 = bbox
                        box_width = max(0, x2 - x1)
                        box_height = max(0, y2 - y1)
                        
                        if box_width > 0 and box_height > 0:
                            total_size += (box_width + box_height) / 2
                            valid_detections += 1
                    except (KeyError, TypeError, ValueError) as e:
                        logger.debug(f"Skipping invalid detection: {e}")
                        continue
                
                if valid_detections > 0:
                    avg_box_size = total_size / valid_detections
                    
                    # Scale kernel size based on average object size
                    kernel_radius = int(np.clip(avg_box_size * 0.8, 
                                               self.min_kernel_size, 
                                               self.max_kernel_size))
                    kernel_radius = max(15, kernel_radius)
                    logger.debug(f"Adaptive kernel: avg={avg_box_size:.1f}, radius={kernel_radius}")
                else:
                    kernel_radius = self.kernel_size
            else:
                kernel_radius = self.kernel_size
            
            # Add Gaussian blobs at each detection center with validation
            for det in detections:
                try:
                    center = det.get('center', [])
                    bbox = det.get('bbox', [])
                    
                    if len(center) != 2 or len(bbox) != 4:
                        continue
                    
                    cx, cy = center
                    
                    # Validate center coordinates
                    if not (0 <= cx < w and 0 <= cy < h):
                        logger.debug(f"Skipping out-of-bounds detection at ({cx}, {cy})")
                        continue
                    
                    # Get detection-specific size for better adaptation
                    if self.adaptive:
                        x1, y1, x2, y2 = bbox
                        det_width = max(0, x2 - x1)
                        det_height = max(0, y2 - y1)
                        det_size = (det_width + det_height) / 2
                        
                        if det_size <= 0:
                            det_kernel = kernel_radius
                        else:
                            det_kernel = int(np.clip(det_size * 0.8, 
                                                    self.min_kernel_size, 
                                                    self.max_kernel_size))
                            det_kernel = max(15, det_kernel)
                    else:
                        det_kernel = kernel_radius
                    
                    # Calculate ROI bounds with proper clamping
                    y_min = max(0, cy - det_kernel)
                    y_max = min(h, cy + det_kernel)
                    x_min = max(0, cx - det_kernel)
                    x_max = min(w, cx + det_kernel)
                    
                    # Validate ROI dimensions
                    kernel_height = y_max - y_min
                    kernel_width = x_max - x_min
                    
                    if kernel_height <= 0 or kernel_width <= 0:
                        continue
                    
                    # Create 2D Gaussian with bounds checking
                    y_range = np.arange(y_min, y_max) - cy
                    x_range = np.arange(x_min, x_max) - cx
                    
                    if len(y_range) == 0 or len(x_range) == 0:
                        continue
                    
                    x_grid, y_grid = np.meshgrid(x_range, y_range)
                    
                    # Gaussian formula with adaptive sigma
                    det_sigma = det_kernel * self.blur_strength
                    gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * det_sigma**2))
                    
                    # Use confidence as intensity multiplier for better visualization
                    intensity = det.get('confidence', 1.0)
                    
                    # Add to density map with bounds safety
                    try:
                        density_map[y_min:y_max, x_min:x_max] += gaussian.astype(np.float32) * intensity
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping gaussian placement: {e}")
                        continue
                    
                except (KeyError, TypeError, ValueError, IndexError) as e:
                    logger.debug(f"Error processing detection for heatmap: {e}")
                    continue
            
            # Normalize density map to 0-255
            if density_map.max() > 0:
                density_map = (density_map / density_map.max() * 255).astype(np.uint8)
            else:
                density_map = density_map.astype(np.uint8)
            
            # Apply single Gaussian blur for smooth appearance (removed double blur)
            blur_size = max(11, min(21, kernel_radius // 4))  # Adaptive blur size
            if blur_size % 2 == 0:
                blur_size += 1  # Must be odd
            density_map = cv2.GaussianBlur(density_map, (blur_size, blur_size), 0)
            
            # Apply colormap (REQ-5: color map representing concentration)
            heatmap_colored = cv2.applyColorMap(density_map, self.colormap)
            
            # Overlay heatmap on original frame
            heatmap_overlay = cv2.addWeighted(
                frame, 
                1 - self.alpha, 
                heatmap_colored, 
                self.alpha, 
                0
            )
            
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            
            # Check performance constraint (SRS: ≤ 1.5s per frame)
            if generation_time > self.config['constraints']['max_heatmap_delay']:
                logger.warning(
                    f"Heatmap generation exceeded constraint: "
                    f"{generation_time:.3f}s > {self.config['constraints']['max_heatmap_delay']}s"
                )
            
            return heatmap_overlay, generation_time
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return frame.copy(), time.time() - start_time
    
    def generate_density_grid(self, frame: np.ndarray, detections: List[Dict], 
                             grid_size: int = 50) -> np.ndarray:
        """
        Generate grid-based density visualization (alternative method)
        
        Args:
            frame: Input frame
            detections: List of detections
            grid_size: Size of each grid cell
        
        Returns:
            grid_overlay: Frame with grid density overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Create grid
        grid_h = h // grid_size + 1
        grid_w = w // grid_size + 1
        density_grid = np.zeros((grid_h, grid_w), dtype=int)
        
        # Count detections in each grid cell
        for det in detections:
            cx, cy = det['center']
            grid_x = min(cx // grid_size, grid_w - 1)
            grid_y = min(cy // grid_size, grid_h - 1)
            density_grid[grid_y, grid_x] += 1
        
        # Draw grid with color intensity based on density
        max_density = density_grid.max() if density_grid.max() > 0 else 1
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                if density_grid[gy, gx] > 0:
                    x1 = gx * grid_size
                    y1 = gy * grid_size
                    x2 = min(x1 + grid_size, w)
                    y2 = min(y1 + grid_size, h)
                    
                    # Color intensity based on density
                    intensity = int(255 * (density_grid[gy, gx] / max_density))
                    color = (0, intensity, 255 - intensity)  # Blue to red
                    
                    # Draw semi-transparent rectangle
                    sub_img = overlay[y1:y2, x1:x2]
                    rect = np.full_like(sub_img, color, dtype=np.uint8)
                    overlay[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 0.7, rect, 0.3, 0)
        
        return overlay
    
    def get_statistics(self) -> Dict:
        """Get heatmap generation statistics"""
        if not self.generation_times:
            return {
                'avg_generation_time': 0.0,
                'total_heatmaps': 0
            }
        
        return {
            'avg_generation_time': np.mean(self.generation_times[-100:]),
            'total_heatmaps': len(self.generation_times),
            'max_generation_time': max(self.generation_times),
            'min_generation_time': min(self.generation_times)
        }
