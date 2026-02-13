"""Scenario 2: Learning by Demonstration

Robot learn and replay continuous movement paths:
1. FIST: Start recording path (robot follows hand continuously)
2. OPEN: Stop recording and replay the learned path(loop)
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


class LearningByDemonstration:
    
    # States
    STATE_WAITING = "WAITING"       # Waiting for user to start
    STATE_RECORDING = "RECORDING"   # Recording path while following hand
    STATE_REPLAYING = "REPLAYING"   # Replaying learned path
    
    def __init__(self):
        #Initialize
        self.vision = VisionController(use_filter=True)
        self.sim = PandaSimulator()
        self.ik_solver = IKSolver(damping=0.01, max_iter=50, tolerance=1e-3, step_size=0.3)
        
        # Share data between vision and sim
        self.hand_pos = None
        self.hand_detected = False
        self.gesture = 'unknown'
        self.running = False
        self.lock = threading.Lock()
        
        # State and path
        self.state = self.STATE_WAITING
        self.path = []  # store trajectory
        self.replay_index = 0 
        
        # Threading
        self.vision_thread = None
        self.physics_thread = None
    
    #Vision thread: track hand pos and gesture
    def vision_loop(self):
        print("[Vision Thread] Started")
        
        while self.running:
            result = self.vision.process_frame()  #see vision controller.
            
            if result:
                with self.lock:
                    if result['hand_detected']:
                        self.hand_detected = True
                        self.hand_pos = np.array(result['hand_center_3d'])  #get 3d hand pos shared with physics thread
                        self.gesture = result['gesture']
                    else:
                        self.hand_detected = False
                        self.hand_pos = None
                
                # Display info
                frame = result['frame']
                
                # State
                state_colors = {
                    self.STATE_WAITING: (255, 255, 0),
                    self.STATE_RECORDING: (0, 0, 255),
                    self.STATE_REPLAYING: (255, 0, 255)
                }
                state_color = state_colors.get(self.state, (255, 255, 255))
                cv2.putText(frame, f"State: {self.state}", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                # Path length
                cv2.putText(frame, f"Path points: {len(self.path)}", (10, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Instructions on frame
                if self.state == self.STATE_WAITING:
                    instruction = "Make FIST to start recording path"
                elif self.state == self.STATE_RECORDING:
                    instruction = "Move hand to record | OPEN to stop & replay"
                elif self.state == self.STATE_REPLAYING:
                    instruction = f"Replaying... ({self.replay_index}/{len(self.path)}) | FIST to re-record"
                
                cv2.putText(frame, instruction, (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Learning by Demonstration', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            time.sleep(0.01)
        
        print("[Vision Thread] Stopped")
    
    #Physics thread: replay learned move
    def physics_loop(self):
        print("[Physics Thread] Started")
        
        target_pos = np.array([0.4, 0.0, 0.3])
        
        while self.running and self.sim.is_viewer_running():
            # Get shared data
            with self.lock:
                hand_detected = self.hand_detected
                hand_pos = self.hand_pos.copy() if self.hand_pos is not None else None
                gesture = self.gesture
            
            current_time = time.time()
            
            if self.state == self.STATE_WAITING:
                # Wait for fist to start recording
                if hand_detected and gesture == 'fist':
                    print("[State] WAITING -> RECORDING")
                    self.state = self.STATE_RECORDING
                    self.path = []  # Clear old path
            
            elif self.state == self.STATE_RECORDING:
                # Record path while following hand
                if hand_detected and hand_pos is not None:
                    target_pos = hand_pos #follow hand pos and record
                    
                    # Record position (every frame)
                    self.path.append(hand_pos.copy())
                    
                    # open hand --> stop recording
                    if gesture == 'open':
                        if len(self.path) >= 10:  # Need at least 10 points
                            print(f"[State] RECORDING -> REPLAYING ({len(self.path)} points)")
                            self.state = self.STATE_REPLAYING
                            self.replay_index = 0
                        else:
                            print(f"[State] Path too short ({len(self.path)} points), back to WAITING")
                            self.state = self.STATE_WAITING
                else:
                    # Lost tracking, back to waiting
                    print("[State] RECORDING -> WAITING (hand lost)")
                    self.state = self.STATE_WAITING
            
            elif self.state == self.STATE_REPLAYING:
                # Replay the recorded path
                if len(self.path) > 0:
                    # Current target from path
                    target_pos = self.path[self.replay_index]
                    
                    # Check if close enough to move to next point
                    ee_pos = self.sim.get_ee_position("ee_site")
                    distance = np.linalg.norm(ee_pos - target_pos)
                    
                    if distance < 0.02:    # --> Move to next point
                        self.replay_index = (self.replay_index + 1) % len(self.path)  #last point --> back to first point
                
                # Reset to record new path
                if hand_detected and gesture == 'fist':
                    print("[State] REPLAYING -> RECORDING (re-record)")
                    self.state = self.STATE_RECORDING
                    self.path = []
            
            # Solve IK and move robot
            joint_positions, converged = self.ik_solver.compute_ik(
                self.sim.model,
                self.sim.data,
                target_pos,
                target_quat=None,
                site_name="ee_site"
            )
            
            #set q
            self.sim.set_joint_positions(joint_positions[:7])
            self.sim.step(n_steps=1)
            
            time.sleep(0.01)
        
        self.running = False
        print("[Physics Thread] Stopped")
    
    def run(self):
        """Run the learning scenario."""
        print("=" * 60)
        print("Scenario 2: Learning by Demonstration (Path Recording)")
        print("=" * 60)
        print("\nHow it works:")
        print("  1. Make FIST → Start recording path")
        print("  2. Move hand to draw path (robot follows)")
        print("  3. OPEN hand → Stop recording & replay path")
        print("  4. Make FIST again → Re-record new path")
        print("\nControls:")
        print("  - FIST: Start recording")
        print("  - OPEN: Stop recording and replay")
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
        
        print("Running... Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
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
    scenario = LearningByDemonstration()
    scenario.run()


if __name__ == "__main__":
    main()
