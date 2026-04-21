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

#### **Terminal 2: ROS 2 - Gazebo Bridge**

Establish the communication link for high-level velocity (Twist) commands.

```bash
ros2 run ros_gz_bridge parameter_bridge /X3/gazebo/command/twist@geometry_msgs/msg/Twist]gz.msgs.Twist
```

#### **Terminal 3: Python Feeder Node**

Run the open-loop feeder to send the pre-calculated Week 1 trajectory to the drone blindly.

```bash
python3 open_loop_feeder.py
```

---

## Week 3: Closed-Loop Receding Horizon Control (RHC)

### 1. Objective

The goal of this week is to implement a real-time closed-loop MPC controller that re-plans every timestep using simulated Mocap feedback. Unlike the open-loop feeder from Week 2, the controller senses the current drone state at each step, solves the finite-horizon optimal control problem, and executes only the first computed command before repeating the cycle. This receding-horizon loop allows the drone to reject disturbances and dynamically discover and avoid the obstacle at runtime.

The cost function minimised at each step is:

$$\min_{u_0 \ldots u_{H-1}} \sum_{k=0}^{H-1} \left( \|x_k - x_{\text{goal}}\|_Q^2 + \|u_k\|_R^2 \right) + \|x_H - x_{\text{goal}}\|_P^2$$

subject to $x_{k+1} = Ax_k + Bu_k$ and elliptic obstacle avoidance constraints.

---

### 2. Architecture

The controller is implemented as a ROS 2 node (`rhc_node.py`) with the following structure:

| Component | Detail |
|---|---|
| State vector | $x = [p_x,\ p_y,\ v_x,\ v_y]^\top$ |
| Control input | $u = [a_x,\ a_y]^\top$, bounded $\pm 2.0\ \text{m/s}^2$ |
| Solver | CasADi `Opti` stack with IPOPT backend |
| Obstacle model | Axis-aligned ellipse: $\frac{(p_x - o_x)^2}{r_x^2} + \frac{(p_y - o_y)^2}{r_y^2} + S_k \geq 2.50$ |
| Pose source | `/world/quadcopter/dynamic_pose/info` (TF), with `/world/quadcopter/pose/info` and `/model/X3/odometry` as fallback topics |
| Command topic | `/X3/gazebo/command/twist` |
| Control rate | 5 Hz (`DT = 0.3 s`) |
| Warm-starting | Previous solution used to initialise each new solve |

**State machine:**

```
TAKEOFF  →  RHC  →  HOVER
```

- `TAKEOFF`: Commands a fixed climb velocity (0.5 m/s) until altitude gain ≥ 0.20 m *and* at least 1.5 s have elapsed, preventing lateral commands near the ground.
- `RHC`: Runs the MPC loop. The obstacle position is read live from the TF tree each cycle rather than being hardcoded, satisfying the Dynamic Discovery requirement.
- `HOVER`: Entered once the drone is within `GOAL_TOLERANCE = 0.10 m` of the goal **and** moving slower than `GOAL_STOP_SPEED = 0.03 m/s`. The dual gate prevents false goal declarations on a fast flythrough.

---

### 3. Final Tuned Parameters

```python
# Mission geometry
GOAL_POSITION            = np.array([2.0, 0.0])
DEFAULT_OBSTACLE_CENTER  = np.array([0.0, 0.0])

# MPC horizon
H               = 25          # steps (7.5 s lookahead at DT=0.3)
DT              = 0.3         # s
CONTROL_RATE    = 5.0         # Hz

# Cost matrices  (all divided by sqrt(2) to match assignment reference notation)
# Q = diag([8.0, 4.2, 2.8, 2.8]) / sqrt(2)   — stage state cost
# R = diag([0.70, 0.24])         / sqrt(2)   — stage control cost
# P = diag([50.0, 18.0, 65.0, 65.0]) / sqrt(2) — terminal state cost

# Obstacle keep-out ellipse
OBSTACLE_RADIUS_X           = 0.68   # m  (padded beyond physical wall thickness)
OBSTACLE_RADIUS_Y           = 1.20   # m  (slightly inside physical half-height)
SLACK_PENALTY               = 30000.0
# Constraint: dist_ellipse + S[k] >= 2.50  (150% margin outside the ellipse boundary)

# Forward progress bias
FORWARD_PROGRESS_WEIGHT     = 0.02

# Control output shaping
MAX_CMD_SPEED               = 0.2    # m/s  hard saturation on output velocity
VEL_FILTER_ALPHA            = 0.25   # low-pass coefficient on finite-difference velocity
# alpha (blend factor) = 0.15        # per-tick velocity update fraction

# Goal completion gates
GOAL_TOLERANCE              = 0.10   # m
GOAL_STOP_SPEED             = 0.03   # m/s

# Takeoff
TAKEOFF_CLIMB_CMD           = 0.5    # m/s vertical command
TAKEOFF_MIN_TIME            = 1.5    # s
TAKEOFF_MAX_TIME            = 6.0    # s
TAKEOFF_MIN_ALTITUDE_GAIN   = 0.20   # m

# Gazebo command frame
X3_CMD_X_SIGN = -1.0
X3_CMD_Y_SIGN = -1.0
```

---

### 4. Tuning Log

Key findings are summarised below.

#### Horizon Length H

| H | Lookahead | Behaviour | NLP variables |
|---|---|---|---|
| 15 | 4.5 s | Hard-constraint infeasibility near obstacle; late abrupt turns | 109 |
| 25 ✓ | 7.5 s | Smooth anticipatory arc around obstacle; IPOPT feasible | 179 |

Increasing `H` from 15 to 25 added ~64% more decision variables. The solve time increase was mitigated by warm-starting each step from the previous solution.

#### Q — State Weight Matrix

`Q = diag([Q_x, Q_y, Q_vx, Q_vy]) / sqrt(2)`

| Version | Values | Effect |
|---|---|---|
| Symmetric baseline | `[10, 10, 1, 1]` | Path goes straight through wall; no lateral bias |
| y-heavy | `[8, 20, 1, 1]` | MPC snaps y→0 through the obstacle |
| Final ✓ | `[8.0, 4.2, 2.8, 2.8]` | Lower `Q_y` allows lateral bypass arc; higher velocity weights damp goal oscillation |

#### R — Control Weight Matrix

`R = diag([R_x, R_y]) / sqrt(2)`

| Version | Values | Effect |
|---|---|---|
| Baseline | `[0.1, 0.1]` | Fast but overshoots goal repeatedly |
| Symmetric mid | `[0.5, 0.5]` | Smoother but equal axis suppression |
| Stall | `[1.0, 1.0]` | Hover-in-place cheaper than advancing |
| Final ✓ | `[0.70, 0.24]` | Asymmetric: high `R_x` damps overshoot; lower `R_y` preserves lateral bypass authority |

---

### 5. Simulation Checkpoint



https://github.com/user-attachments/assets/49794d7e-c4d7-4585-82a3-6dd9871a19f1



The drone successfully navigates from `(-2, 0)` to `(2, 0)`, curving around the central wall obstacle without collision, with the goal declared only after both the distance and speed gates are satisfied.

---

### 6. Instructions to Run

Three terminals are required.

#### **Terminal 1: Start Gazebo**

```bash
ign gazebo -r quadcopter.sdf
```

#### **Terminal 2: ROS 2 – Gazebo Bridges**

Three topics must be bridged. Run each `parameter_bridge` command in its own sub-shell, or chain them:

```bash
# Pose feedback (dynamic transforms)
ros2 run ros_gz_bridge parameter_bridge \
  /world/quadcopter/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V

# Static pose (fallback)
ros2 run ros_gz_bridge parameter_bridge \
  /world/quadcopter/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V

# Velocity command
ros2 run ros_gz_bridge parameter_bridge \
  /X3/gazebo/command/twist@geometry_msgs/msg/Twist]gz.msgs.Twist
```


#### **Terminal 3: RHC Controller Node**

```bash
python3 rhc_node.py
```

<!-- 
The node will print tuning log paths on startup and emit timestamped state logs during flight. Two output files are written to the working directory:

| File | Contents |
|---|---|
| `rhc_tuning_log.csv` | Per-tick telemetry: position, velocity, command, goal, obstacle distance, slack, bypass phase |
| `rhc_tuning_params.txt` | Static parameter dump for reproducibility |
-->

#### **Dependencies**

```bash
pip install casadi numpy
# ROS 2 packages: rclpy, geometry_msgs, tf2_msgs, nav_msgs, ros_gz_bridge
```
