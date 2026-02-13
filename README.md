# Human-Robot Interaction Simulation 

Using **Python** and **MuJoCo** physics simulation and **MediaPipe** hand tracking. Control a Franka Emika Panda robot arm using real-time hand gestures captured via webcam.

## Features

- **Robot Simulation**: Franka Emika Panda robot in MuJoCo
- **Real-time Hand Tracking**: MediaPipe-based gesture recognition
- **IK**: Jacobian Inverse method with damping least square
- **Signal Filtering**: OneEuroFilter
- **Three demonstration Scenarios**

## System Architecture

```
project/

├── core/               
│     ├── vision_controller.py # MediaPipe hand tracking & coordinate convert
│     ├── panda_sim.py         # Franka Panda wrapper
│     ├── ik_solver.py         # Inverse kinematics
│     └── scenarios/           # 3 scenarios
│         ├── handover.py
│         ├── learning_by_demonstration.py
│         └── path_following.py
├── utils/          
│     └── filters.py  
├── franka_emika_panda/          # Robot model (from MuJoCo Menagerie)
│     └── scene.xml  
├── test/                        # Test scripts
│     ├── mujoco_test.py         # start and run mujoco a demo program
│     └── mediapipe_test.py      # test webcam and mediapipe
├── main.py                      # Main application
└── README.md                    
```

### Prerequisites

- **Python 3.9 or 3.10**
- **Webcam** 
- **Windows/Linux/macOS**
- **NumPy 1.24.0**
- **OpenCV 4.8.0**
- **MuJoCo 3.0.0**
- **MediaPipe 0.10.0**


### Run the System

**0 : Tests**
```bash
# Test webcam & MediaPipe 
python test/mediapipe_test.py

# Test mujoco & IK solver with a certain trajectory(square)
python test/mujoco_test.py
```

**Option 1: Main Menu**
```bash
python main.py
```
This launches an interactive menu where you can select scenarios.


**Option 2: Directly Run Individual Tests**
```bash
# Scenario 1: Handover cube
python simulation/scenarios/scenario1_handover.py

# Scenario 2: Learning by Demonstration
python simulation/scenarios/scenario2_learning.py

# Scenario 3: Path Following
python simulation/scenarios/scenario3_path_following.py
```


## Scenarios

### 1 Handover cube

You can pass a cube to the robot, and it will move it to a certain position.

- fist gesture → start handover ()
- move your hand → move the cube in the mujoco environment 
- cube get into the workspace → robot come and take the cube 
- open gesture → pass the cube to the robot
- robot move to home position

**Use Case:** Human-robot collaboration, workspace sharing

### 2 Learning by Demonstration

The robot records your hand movements and replays.

- fist gesture → start recording
- move your hand → record your hand movements
- open gesture → stop recording and replay

**Use Case:** Easy robotic programming of repetitive tasks 

### 3️ Path Following

The robot tracks your hand movements in real-time.

- Move hand slowly → Smooth tracking
- Move hand quickly → Tests responsiveness
- Draw 3D paths in the air

**Use Case:** Remote robot manipulation


## Credits

- **Robot Model**: Franka Emika Panda ([from MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie))
- **Hand Tracking**: Google MediaPipe
- **Physics Engine**: MuJoCo by DeepMind
- **Filtering**: OneEuro Filter ((https://gery.casiez.net/1euro/))
- **Claude 4.5**: by Anthropic
   - implement damping least square method to improve Jacobian Inverse method
   - Code optimization (filters.py and vision_controller.py )
   - quaternion (but not used in this project)
   




