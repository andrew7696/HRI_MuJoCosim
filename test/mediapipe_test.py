"""MediaPipe Hand Tracking Test

This standalone script tests the MediaPipe hand tracking functionality.
It opens the webcam, detects hand landmarks, visualizes them on screen,
and prints normalized coordinates to the console.

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vision_controller import VisionController


def main():
    """Run MediaPipe hand tracking test."""
    print("=" * 60)
    print("MediaPipe Hand Tracking Test")
    print("=" * 60)
    print("\nControls:")
    print("  - Move your hand in front of the camera")
    print("  - Make a fist or open palm to test gesture detection")
    print("  - Press 'q' to quit")
    print("\n" + "=" * 60 + "\n")
    
    # Initialize vision controller
    vision = VisionController(use_filter=True)
    
    # Start camera
    if not vision.start_camera():
        print("Failed to start camera. Exiting...")
        return
    
    print("Camera started successfully!")
    print("Waiting for hand detection...\n")
    
    frame_count = 0
    
    try:
        while True:
            # Process frame
            result = vision.process_frame()
            
            if result is None:
                print("Failed to capture frame")
                break
            
            frame = result['frame']
            hand_detected = result['hand_detected']
            
            # Display information
            if hand_detected:
                hand_center_2d = result['hand_center_2d']
                hand_center_3d = result['hand_center_3d']
                gesture = result['gesture']
                
                # Print to console every 10 frames
                if frame_count % 10 == 0:
                    print(f"Frame {frame_count:5d} | "
                          f"2D: ({hand_center_2d[0]:.3f}, {hand_center_2d[1]:.3f}) | "
                          f"3D: ({hand_center_3d[0]:.3f}, {hand_center_3d[1]:.3f}, {hand_center_3d[2]:.3f}) | "
                          f"Gesture: {gesture}")
            else:
                # Display "No hand detected" on frame
                h, w, _ = frame.shape
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            # Display frame
            cv2.imshow('MediaPipe Hand Tracking Test', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        vision.stop_camera()
        cv2.destroyAllWindows()
        print("\nTest completed!")


if __name__ == "__main__":
    main()
