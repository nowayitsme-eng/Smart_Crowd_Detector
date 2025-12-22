"""
Test script to verify resize functionality
"""
import cv2
import numpy as np

# Test different resize factors
RESIZE_FACTORS = [1.0, 0.75, 0.5, 0.33]

# Create a test frame
frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
cv2.putText(frame, "Original Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

print("Original frame shape:", frame.shape)
print("\nTesting resize factors:")
print("-" * 60)

for RESIZE_FACTOR in RESIZE_FACTORS:
    if RESIZE_FACTOR < 1.0:
        new_width = int(frame.shape[1] * RESIZE_FACTOR)
        new_height = int(frame.shape[0] * RESIZE_FACTOR)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate performance improvement
        original_pixels = frame.shape[0] * frame.shape[1]
        resized_pixels = resized.shape[0] * resized.shape[1]
        speedup = original_pixels / resized_pixels
        
        print(f"RESIZE_FACTOR = {RESIZE_FACTOR:.2f}")
        print(f"  New shape: {resized.shape}")
        print(f"  Pixels: {original_pixels:,} → {resized_pixels:,}")
        print(f"  Expected speedup: {speedup:.1f}x")
        print(f"  Memory reduction: {(1 - RESIZE_FACTOR**2) * 100:.1f}%")
        print()
    else:
        print(f"RESIZE_FACTOR = {RESIZE_FACTOR:.2f} (no resize - full resolution)")
        print(f"  Shape: {frame.shape}")
        print()

print("✅ Resize functionality test complete!")
print("\nRecommendations:")
print("  - For balanced performance: RESIZE_FACTOR = 0.5 (4x faster)")
print("  - For maximum speed: RESIZE_FACTOR = 0.33 (9x faster)")
print("  - For best accuracy: RESIZE_FACTOR = 1.0 (no resize)")
