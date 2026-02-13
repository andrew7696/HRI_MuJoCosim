"""Scenario 3: Path Following

Robot ee follow user's hand in real-time, 
similar to the mujoco test but with hand input.
"""

import threading
import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.vision_controller import VisionController
from core.ik_solver import IKSolver
from core.panda_sim import PandaSimulator
import cv2


class PathFollowing:
    
    def __init__(self):
        """Initialize the scenario."""
        self.vision = VisionController(use_filter=True)
        self.sim = PandaSimulator()
        self.ik_solver = IKSolver(damping=0.01, max_iter=50, tolerance=1e-3, step_size=0.3)
        
        # Share data between vision and simulator
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.hand_detected = False
        self.gesture = 'unknown'
        self.gripper_target = 0  
        self.lock = threading.Lock()
        
        # Statistics
        self.tracking_error = 0.0
        
        # Threading
        self.vision_thread = None
        self.physics_thread = None
    
    def vision_loop(self):
        """Vision thread: track hand position."""
        print("[Vision Thread] Started")
        
        while self.running:
            result = self.vision.process_frame()
            
            if result:
                with self.lock:
                    if result['hand_detected']:
                        self.hand_detected = True
                        self.target_pos = np.array(result['hand_center_3d'])
                        self.gesture = result['gesture']
                        
                        # Update gripper based on gesture
                        if self.gesture == 'fist':
                            self.gripper_target = 0.0  # Close
                        elif self.gesture == 'open':
                            self.gripper_target = 255 # Open
                    else:
                        self.hand_detected = False
                
                # Add tracking info to frame
                frame = result['frame']
                if self.tracking_error > 0:
                    error_text = f"Tracking Error: {self.tracking_error*1000:.1f}mm"
                    cv2.putText(
                        frame,
                        error_text,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                
                # Display gripper status
                with self.lock:
                    gripper_status = "CLOSED" if self.gripper_target == 0.0 else "OPEN"
                    gripper_color = (0, 0, 255) if self.gripper_target == 0.0 else (0, 255, 0)
                
                gripper_text = f"Gripper: {gripper_status}"
                cv2.putText(
                    frame,
                    gripper_text,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    gripper_color,
                    2
                )
                
                cv2.imshow('Path Following - Hand Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            time.sleep(0.01)
        
        print("[Vision Thread] Stopped")

    #Physics thread:robot follow hand 
    def physics_loop(self):
        print("[Physics Thread] Started")
        
        last_print = time.time()
        
        while self.running and self.sim.is_viewer_running():
            # Get shared data
            with self.lock:
                target = self.target_pos.copy()
                detected = self.hand_detected
                gripper = self.gripper_target
            
            if detected:
                # Solve IK to follow hand
                joint_positions, converged = self.ik_solver.compute_ik(
                    self.sim.model,
                    self.sim.data,
                    target,
                    target_quat=None,
                    site_name="ee_site"
                )
                
                # Set q
                self.sim.set_joint_positions(joint_positions[:7])
                
                # Control gripper
                self.sim.set_gripper(gripper)
                
                # Calculate tracking error
                ee_pos = self.sim.get_ee_position("ee_site")
                error = np.linalg.norm(ee_pos - target)
                
                with self.lock:
                    self.tracking_error = error
                
                # Print status periodically
                current_time = time.time()
                if current_time - last_print >= 1.0:
                    gripper_status = "CLOSED" if gripper == 0.0 else "OPEN"
                    print(f"Target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] | "
                          f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                          f"Error: {error*1000:.1f}mm | "
                          f"IK: {'OK' if converged else 'FAIL'} | "
                          f"Gripper: {gripper_status}")
                    last_print = current_time
            
            # Step simulation
            self.sim.step(n_steps=1)
            
            time.sleep(0.01)
        
        self.running = False
        print("[Physics Thread] Stopped")
    
    def run(self):
        """Run the path following scenario."""
        print("=" * 60)
        print("Scenario 3: Path Following")
        print("=" * 60)
        print("\nBehavior:")
        print("  - Robot ee follows your hand ")
        print("  - Move hand to draw paths in 3D space")
        print("  - Gripper responds to hand gestures:")
        print("    * Fist  → Gripper CLOSES")
        print("    * Open hand  → Gripper OPENS")
        print("\nControls:")
        print("  - Move hand slowly for smooth tracking")
        print("  - Press 'q' in camera window to quit")
        print("=" * 60 + "\n")
        
        # Initialize
        if not self.vision.start_camera():
            print("Failed to start camera")
            return
        
        if not self.sim.load_model():
            print("Failed to load model")
            return
        
        if not self.sim.start_viewer():
            print("Failed to start viewer")
            return
        
        # Start threads
        self.running = True
        
        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        self.physics_thread = threading.Thread(target=self.physics_loop, daemon=True)
        
        self.vision_thread.start()
        self.physics_thread.start()
        
        print("running... Press Ctrl+C to stop\n")
        print("Waiting for hand detection...\n")
        
        try:
            # Wait for threads to finish
            while self.running:
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.running = False
            
            if self.vision_thread:
                self.vision_thread.join(timeout=1.0)
            if self.physics_thread:
                self.physics_thread.join(timeout=1.0)
            
            self.vision.stop_camera()
            cv2.destroyAllWindows()
            self.sim.close()
            
            print("\nGoodbye!")


def main():
    scenario = PathFollowing()
    scenario.run()


if __name__ == "__main__":
    main()
