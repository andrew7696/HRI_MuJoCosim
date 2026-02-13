"""Main orchestration for this project.

This script provides a entrance to select and run different scenarios:
1. Handover Coordination
2. Learning by Demonstration
3. Path Following
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.scenarios import (
    HandoverCoordination,
    LearningByDemonstration,
    PathFollowing
)


def print_menu():
    """Print main menu."""
    print("\nSelect a scenario:")
    print("  1. Handover Coordination")
    print("     - Robot hands over object to you")
    print("     - Face-to-face interaction | Open hand to receive")
    print()
    print("  2. Learning by Demonstration")
    print("     - Record trajectory with fist gesture")
    print("     - Robot replays your movements")
    print()
    print("  3. Path Following")
    print("     - Robot follows hand in real-time")
    print("     - Gesture control for gripper")
    print()
    print("  0. Exit")
    print()


def get_user_choice() -> int:
    """Get user's menu choice.
    Returns:
        Selected option number (0-3)
    """
    while True:
        try:
            choice = input("Enter your choice (0-3): ").strip()
            choice_int = int(choice)
            
            if 0 <= choice_int <= 3:
                return choice_int
            else:
                print("Invalid choice. Please enter a number between 0 and 3.")
        
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return 0

#user enter choice
def run_scenario(choice: int) -> None:

    if choice == 1:
        scenario = HandoverCoordination()
        scenario.run()
    
    elif choice == 2:
        scenario = LearningByDemonstration()
        scenario.run()
    
    elif choice == 3:
        scenario = PathFollowing()
        scenario.run()


def main():
    """entrance."""
    print("\n" + "=" * 60)
    print("SYSTEM REQUIREMENTS:")
    print("  - Webcam connected and working")
    print("  - Python packages: mujoco, mediapipe, opencv-python, numpy")
    print("  - Franka Panda model files in franka_emika_panda/")
    print("=" * 60)
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == 0:
            print("Goodbye!\n")
            break
        
        try:
            run_scenario(choice)
        
        except Exception as e:
            print(f"\nError running scenario: {e}")
            import traceback
            traceback.print_exc()
            print("\nReturning to main menu...")
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
