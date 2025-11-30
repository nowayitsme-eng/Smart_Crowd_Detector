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
    
    def generate_heatmap(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, float]:
        """
        Generate crowd density heatmap
        
        Args:
            frame: Input frame
            detections: List of detections with center points
        
        Returns:
            heatmap_overlay: Frame with heatmap overlay
            generation_time: Time taken to generate heatmap
        
        Implements:
            - REQ-4: Generate localized density zones
            - REQ-5: Apply color map for crowd concentration
        """
        start_time = time.time()
        
        if not self.enabled or len(detections) == 0:
            generation_time = time.time() - start_time
            return frame.copy(), generation_time
        
        try:
            h, w = frame.shape[:2]
            
            # Create empty density map (REQ-4: localized density zones)
            density_map = np.zeros((h, w), dtype=np.float32)
            
            # Add Gaussian blobs at each detection center
            for det in detections:
                cx, cy = det['center']
                
                # Ensure center is within bounds
                cx = max(0, min(w - 1, cx))
                cy = max(0, min(h - 1, cy))
                
                # Create small Gaussian kernel around detection
                y_min = max(0, cy - self.kernel_size)
                y_max = min(h, cy + self.kernel_size)
                x_min = max(0, cx - self.kernel_size)
                x_max = min(w, cx + self.kernel_size)
                
                # Add Gaussian contribution
                y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(
                    -(((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * (self.kernel_size / 3) ** 2))
                )
                
                density_map[y_min:y_max, x_min:x_max] += gaussian
            
            # Normalize density map to 0-255
            if density_map.max() > 0:
                density_map = (density_map / density_map.max() * 255).astype(np.uint8)
            else:
                density_map = density_map.astype(np.uint8)
            
            # Apply Gaussian blur for smoother appearance
            density_map = cv2.GaussianBlur(density_map, (21, 21), 0)
            
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
