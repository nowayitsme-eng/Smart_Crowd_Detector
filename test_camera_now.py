"""Test camera availability"""
import cv2

print("Testing camera access...")
print("\nTrying different camera indices and backends:\n")

for i in range(3):
    print(f"Camera {i}:")
    
    # Try DirectShow (Windows)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ DirectShow backend - Working (Frame: {frame.shape})")
        else:
            print(f"  ✗ DirectShow backend - Opened but no frames")
        cap.release()
    else:
        print(f"  ✗ DirectShow backend - Failed")
    
    # Try default backend
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Default backend - Working (Frame: {frame.shape})")
        else:
            print(f"  ✗ Default backend - Opened but no frames")
        cap.release()
    else:
        print(f"  ✗ Default backend - Failed")
    
    print()

print("\nTest complete!")
