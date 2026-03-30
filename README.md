# SC 627: Motion Planning & Coordination - Assignment 2
## Week 2: Gazebo Integration & Open-Loop Execution

### 1. Objective
The goal of this week is to transition from the mathematical optimization developed in Week 1 to a simulated hardware environment. This stage demonstrates the "Simulation Gap" and the necessity of feedback control due to accumulated drift and model mismatch.

### 2. Failure Insight Log
As per the assignment requirements, the following observations were made during the open-loop flight:

* **Accumulated Drift**: Because the controller provides no position feedback, small integration errors in Gazebo and micro-latencies in motor response caused the drone to veer off the planned path.
* **Model Mismatch**: The Week 1 optimizer assumes a simplified double-integrator model. In reality, the X3 UAV must physically tilt to accelerate, creating a delay that the open-loop feeder cannot compensate for.
* **Result**: The drone successfully took off but eventually drifted into the arena wall instead of reaching the goal table.

### 3. Simulation Video
The following video documents the open-loop flight attempt:


https://github.com/user-attachments/assets/c28869fb-12b6-480b-b641-4c089d4558bb


---

### 4. Instructions to Run
To replicate the simulation, follow the sequence below using three separate terminals.

#### **Terminal 1: Start Gazebo**
Launch the simulation environment. The `-r` flag ensures the physics engine starts running immediately.
```bash
ign gazebo -r quadcopter.sdf
```

#### **Terminal 2: Start Gazebo**
Establish the communication link for high-level velocity (Twist) commands.
```bash
ros2 run ros_gz_bridge parameter_bridge /X3/gazebo/command/twist@geometry_msgs/msg/Twist]gz.msgs.Twist
```

#### **Terminal 3: Python Feeder Node**
Run the open-loop feeder to send the pre-calculated Week 1 trajectory to the drone blindly.
```bash
python3 open_loop_feeder.py
```
