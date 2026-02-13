"""MuJoCo IK Solver Test

This script tests the inverse kinematics solver by making the robot
end-effector follow a red dot that moves in a square trajectory.

This verifies the IK solver and MuJoCo integration without camera input.
Press ESC to quit.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.panda_sim import PandaSimulator
from core.ik_solver import IKSolver


def generate_square_trajectory(t: float, center: np.ndarray, size: float = 0.2, period: float = 10.0) -> np.ndarray:
    """Generate a square trajectory.
    
    Args:
        t: Current time in seconds
        center: Center position [x, y, z]
        size: Size of the square (side length)
        period: Time to complete one square (seconds)
        
    Returns:
        Current target position [x, y, z]
    """
    # Normalize time to [0, 1] within period
    phase = (t % period) / period
    
    # Define square corners (clockwise)
    half_size = size / 2.0
    corners = np.array([
        [center[0] + half_size, center[1] + half_size, center[2]],  # Top-right
        [center[0] + half_size, center[1] - half_size, center[2]],  # Bottom-right
        [center[0] - half_size, center[1] - half_size, center[2]],  # Bottom-left
        [center[0] - half_size, center[1] + half_size, center[2]],  # Top-left
    ])
    
    # Determine which edge we're on
    segment = int(phase * 4)
    segment_phase = (phase * 4) % 1.0
    
    # Interpolate between corners
    start_corner = corners[segment]
    end_corner = corners[(segment + 1) % 4]
    
    position = start_corner + segment_phase * (end_corner - start_corner)
    
    return position


def main():
    """Run MuJoCo IK test with square trajectory."""
    print("=" * 60)
    print("MuJoCo IK Solver Test - Square Trajectory")
    print("=" * 60)
    print("\nThis test will:")
    print("  1. Load the Panda robot in MuJoCo")
    print("  2. Generate a square trajectory (red dot)")
    print("  3. Use IK to make the robot follow the red dot")
    print("\nPress ESC in the viewer window to quit")
    print("=" * 60 + "\n")
    
    # Initialize simulator
    sim = PandaSimulator()
    
    if not sim.load_model():
        print("Failed to load model. Exiting...")
        return
    
    if not sim.start_viewer():
        print("Failed to start viewer. Exiting...")
        return
    
    # Initialize IK solver
    ik_solver = IKSolver(
        damping=0.01,
        max_iter=50,
        tolerance=1e-3,
        step_size=0.3
    )
    
    # Square trajectory parameters
    center = np.array([0.4, 0.0, 0.3])  # Center of square
    size = 0.2  # Square side length
    period = 12.0  # Time for one complete square
    
    print(f"Square trajectory:")
    print(f"  Center: {center}")
    print(f"  Size: {size}m x {size}m")
    print(f"  Period: {period}s")
    print("\nStarting simulation...\n")
    
    # Simulation loop
    start_time = time.time()
    last_print_time = start_time
    iteration = 0
    
    # First, create an end-effector site if it doesn't exist
    # We'll need to modify the XML to add this site, but for now we'll use the hand body
    
    try:
        while sim.is_viewer_running():
            current_time = time.time() - start_time
            
            # Generate target position
            target_pos = generate_square_trajectory(current_time, center, size, period)
            
            # Visualize target (this will just print if site doesn't exist)
            sim.update_target_position(target_pos, "target")
            
            # Solve IK
            joint_positions, converged = ik_solver.compute_ik(
                sim.model,
                sim.data,
                target_pos,
                target_quat=None,  # Position-only IK
                site_name="ee_site"
            )
            
            # Set joint positions (first 7 joints for arm)
            sim.set_joint_positions(joint_positions[:7])
            
            # Step simulation
            sim.step(n_steps=1)
            
            # Print status every second
            if current_time - last_print_time >= 1.0:
                ee_pos = sim.get_ee_position("ee_site")
                error = np.linalg.norm(ee_pos - target_pos)
                
                print(f"Time: {current_time:6.2f}s | "
                      f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] | "
                      f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                      f"Error: {error:.4f}m | "
                      f"Converged: {converged}")
                
                last_print_time = current_time
            
            iteration += 1
            
            # Limit frame rate
            time.sleep(0.01)  # ~100 Hz control loop
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        sim.close()
        print("\nTest completed!")
        print(f"Total iterations: {iteration}")
        print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
