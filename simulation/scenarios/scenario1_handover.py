"""Scenario 1: Handover Coordination 

Human handover object to robot:
1. FIST → Blue cube appears and follows hand (It represents human holding object)
2. User moves cube into robot workspace (there's an assigned workspace for robot to grab the cube)
3. Robot extends and grabs the cube
4. User OPENS hand → Robot takes cube back to home position
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


class HandoverCoordination:
    
    # States
    STATE_WAITING = "WAITING"           # waiting for fist gesture
    STATE_USER_HOLDING = "USER_HOLDING" # user holding cube with fist
    STATE_EXTENDING = "EXTENDING"       # extending to grab cube
    STATE_GRASPING = "GRASPING"         # closing gripper on cube
    STATE_RETRACTING = "RETRACTING"     # taking cube back to home
    
    # Positions
    HOME_POS = np.array([0.3, 0.0, 0.4])
    
    #Assigned workspace bounds for cube grabbing
    WORKSPACE_X_MIN = 0.35
    WORKSPACE_X_MAX = 0.65
    WORKSPACE_Y_MIN = -0.25
    WORKSPACE_Y_MAX = 0.25
    WORKSPACE_Z_MIN = 0.15
    WORKSPACE_Z_MAX = 0.45
    
    def __init__(self):
        #Initialize 
        self.vision = VisionController(use_filter=True, face_to_face=True)
        self.sim = PandaSimulator()
        self.ik_solver = IKSolver(damping=0.01, max_iter=50, tolerance=1e-3, step_size=0.3)
        
        # Share data between vision and simulator
        self.hand_pos = None
        self.hand_detected = False
        self.gesture = 'unknown'
        self.running = False    # --> control both threads(vision and sim)
        self.lock = threading.Lock()  # --> sim read after vision write completed
        
        # Initial state
        self.state = self.STATE_WAITING   
        self.target_pos = self.HOME_POS.copy()
        self.gripper_target = 255.0  

        # Cube tracking
        self.cube_pos = None  # No cube shown before user fist
        self.cube_body_id = None  #see scene.xml line 23(set after fist)
        
        # Timing
        self.state_start_time = time.time()
        
        # Threading
        self.vision_thread = None
        self.physics_thread = None
    
    #if position is in robot workspace
    def is_in_workspace(self, pos):

        if pos is None:
            return False
        return (self.WORKSPACE_X_MIN <= pos[0] <= self.WORKSPACE_X_MAX and
                self.WORKSPACE_Y_MIN <= pos[1] <= self.WORKSPACE_Y_MAX and
                self.WORKSPACE_Z_MIN <= pos[2] <= self.WORKSPACE_Z_MAX)
    
    #Vision thread: track hand position and gesture
    def vision_loop(self):
        print("[Vision Thread] Started")
        
        while self.running:    # --> duothread controller
            result = self.vision.process_frame()  # --> see vision_controller.py process_frame
            
            if result:
                with self.lock:
                    # --> lock
                    if result['hand_detected']:   # --> share with physics
                        self.hand_detected = True
                        self.hand_pos = np.array(result['hand_center_3d'])
                        self.gesture = result['gesture']
                    else:
                        self.hand_detected = False
                        self.hand_pos = None
                        self.gesture = 'unknown'
                # --> release 
                
                # Add state info to frame
                frame = result['frame']
                
                # Display state on frame
                state_text = f"State: {self.state}"
                state_colors = {
                    self.STATE_WAITING: (255, 255, 0),
                    self.STATE_USER_HOLDING: (0, 255, 255),
                    self.STATE_EXTENDING: (255, 165, 0),
                    self.STATE_GRASPING: (0, 255, 0),
                    self.STATE_RETRACTING: (0, 0, 255)
                }
                state_color = state_colors.get(self.state, (255, 255, 255))
                cv2.putText(frame, state_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                # Display instructions on frame
                if self.state == self.STATE_WAITING:
                    instruction = "Make FIST to create cube"
                elif self.state == self.STATE_USER_HOLDING:
                    in_workspace = self.is_in_workspace(self.hand_pos) if self.hand_pos is not None else False
                    if in_workspace:
                        instruction = "Cube in workspace! Keep FIST..."
                    else:
                        instruction = "Move cube into robot workspace"
                elif self.state == self.STATE_EXTENDING:
                    instruction = "Robot extending to grab..."
                elif self.state == self.STATE_GRASPING:
                    instruction = "OPEN hand to release"
                elif self.state == self.STATE_RETRACTING:
                    instruction = "Robot taking cube home..."
                
                cv2.putText(frame, instruction, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show workspace status on frame
                if self.hand_pos is not None:
                    in_ws = self.is_in_workspace(self.hand_pos)
                    ws_text = "In Workspace: " + ("YES" if in_ws else "NO")
                    ws_color = (0, 255, 0) if in_ws else (0, 0, 255)
                    cv2.putText(frame, ws_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ws_color, 2)
            
                cv2.imshow('Handover Coordination - Hand Tracking', frame)
                # press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    self.running = False
            
            time.sleep(0.01)
        
        print("[Vision Thread] Stopped")
    
    #Physics thread: control robot and show cube in viewer
    def physics_loop(self):
        print("[Physics Thread] Started")
        
        #find cube that user controls
        try:
            self.cube_body_id = self.sim.model.body('handover_cube').id            
            cube_body = self.sim.model.body('handover_cube')
            # Find freejoint so we can move the cube
            self.cube_joint_id = None
            for i in range(self.sim.model.njnt):
                if self.sim.model.jnt_bodyid[i] == self.cube_body_id:
                    self.cube_joint_id = i
                    break
            
            if self.cube_joint_id is not None:
                self.cube_qpos_adr = self.sim.model.jnt_qposadr[self.cube_joint_id] # qpos address of the cube(cube_qpos_adr = 9 0~8 is for the robot)
                print(f"[Cube] Found cube joint {self.cube_joint_id}, qpos address: {self.cube_qpos_adr}")
            else:
                print("[Warning] Cube joint not found")
        except Exception as e: 
            print(f"[Warning] Cube setup failed: {e}")
            self.cube_body_id = None
            self.cube_joint_id = None
        
        while self.running and self.sim.is_viewer_running():
            # Get shared data from vision thread
            with self.lock:
                hand_detected = self.hand_detected
                hand_pos = self.hand_pos.copy() if self.hand_pos is not None else None
                gesture = self.gesture
            
            current_time = time.time()
            state_duration = current_time - self.state_start_time # -->to stabilize state transitions 
            
            if self.state == self.STATE_WAITING:
                # Wait for fist
                self.target_pos = self.HOME_POS
                self.gripper_target = 255.0  # Gripper open
                
                # Hide cube (move away to a far position)
                if self.cube_joint_id is not None:
                    self.sim.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr+7] = [5.0, 0, -1, 1, 0, 0, 0]
                
                if hand_detected and gesture == 'fist':
                    print(f"[State] WAITING -> USER_HOLDING (fist detected)")
                    self.state = self.STATE_USER_HOLDING 
                    self.state_start_time = current_time
            
            elif self.state == self.STATE_USER_HOLDING:
                # Move cube to follow hand position
                if hand_pos is not None and self.cube_joint_id is not None:
                    # Update cube pos
                    self.sim.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr+3] = hand_pos
                    self.sim.data.qpos[self.cube_qpos_adr+3:self.cube_qpos_adr+7] = [1, 0, 0, 0]  # I didn't want it to rotate 
                    self.cube_pos = hand_pos.copy()
                    
                    # Check if cube is in workspace
                    if self.is_in_workspace(hand_pos):
                        # Wait a for a while (make sure user is stable)
                        if state_duration > 1.0:
                            print(f"[State] USER_HOLDING -> EXTENDING (cube in workspace)")
                            self.state = self.STATE_EXTENDING
                            self.state_start_time = current_time
                
                # If user releases (opens hand), back to WAITING
                if gesture == 'open' or not hand_detected:
                    print(f"[State] USER_HOLDING -> WAITING (hand released)")
                    self.state = self.STATE_WAITING
                    self.state_start_time = current_time
            
            elif self.state == self.STATE_EXTENDING:
                # Robot moves to cube pos
                if self.cube_pos is not None:
                    # Offset slightly above cube for grasping
                    self.target_pos = self.cube_pos + np.array([0, 0, 0.05])
                    self.gripper_target = 255.0  # Keep gripper open
                    
                    # reached pos?
                    ee_pos = self.sim.get_ee_position("ee_site")
                    distance = np.linalg.norm(ee_pos - self.target_pos)
                    
                    if distance < 0.03 or state_duration > 3.0:
                        print(f"[State] EXTENDING -> GRASPING (position reached)")
                        self.state = self.STATE_GRASPING
                        self.state_start_time = current_time
            
            elif self.state == self.STATE_GRASPING:
                # Robot grasping cube, waiting for user to release
                self.gripper_target = 0.0  # Close gripper
                
                # Wait for user to open hand
                if gesture == 'open':
                    print(f"[State] GRASPING -> RETRACTING (hand opened, taking cube)")
                    self.state = self.STATE_RETRACTING
                    self.state_start_time = current_time
                
                # Timeout
                if state_duration > 10.0:
                    print(f"[State] GRASPING -> RETRACTING (timeout)")
                    self.state = self.STATE_RETRACTING
                    self.state_start_time = current_time
            
            elif self.state == self.STATE_RETRACTING:
                # Return cube to home
                self.target_pos = self.HOME_POS
                self.gripper_target = 0.0  # Keep gripper closed
                
                # Move cube with robot (simply stick cube to the end effector) 
                if self.cube_joint_id is not None:
                    ee_pos = self.sim.get_ee_position("ee_site")
                    self.sim.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr+3] = ee_pos
                
                # reached home?
                ee_pos = self.sim.get_ee_position("ee_site")
                distance = np.linalg.norm(ee_pos - self.HOME_POS)
                
                if distance < 0.02 or state_duration > 3.0:
                    print(f"[State] RETRACTING -> WAITING (home reached)")
                    self.state = self.STATE_WAITING
                    self.state_start_time = current_time
                    self.gripper_target = 255.0  # Open gripper
            
            # Solve IK and move robot
            joint_positions, converged = self.ik_solver.compute_ik(
                self.sim.model,
                self.sim.data,
                self.target_pos,
                target_quat=None,
                site_name="ee_site"
            )
            
            # Set joint pos and gripper
            self.sim.set_joint_positions(joint_positions[:7])
            self.sim.set_gripper(self.gripper_target)
            
            # Step simulation
            self.sim.step(n_steps=1)
            
            time.sleep(0.01)
        
        self.running = False
        print("[Physics Thread] Stopped")
    
    def run(self):
        print("=" * 60)
        print("Scenario 1: Handover (Human to Robot)")
        print("=" * 60)
        print("\nBehavior:")
        print("  1. Make FIST → Blue cube appears and follows your hand")
        print("  2. Move cube into robot workspace")
        print("  3. Robot extends and grabs the cube")
        print("  4. OPEN hand → Robot takes cube back to home")
        print("\nWorkspace Bounds:")
        print(f"  X: {self.WORKSPACE_X_MIN:.2f} to {self.WORKSPACE_X_MAX:.2f}")
        print(f"  Y: {self.WORKSPACE_Y_MIN:.2f} to {self.WORKSPACE_Y_MAX:.2f}")
        print(f"  Z: {self.WORKSPACE_Z_MIN:.2f} to {self.WORKSPACE_Z_MAX:.2f}")
        print("\nControls Guide:")
        print("  - FIST: Create and hold cube")
        print("  - Move hand: Move cube")
        print("  - OPEN hand: Release cube to robot")
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
    scenario = HandoverCoordination()
    scenario.run()


if __name__ == "__main__":
    main()
