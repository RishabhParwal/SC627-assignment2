import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
import casadi as ca
import numpy as np
import time
import csv
import os
import math


GOAL_POSITION = np.array([2.0, 0.0])
DEFAULT_OBSTACLE_CENTER = np.array([0.0, 0.0])
OBSTACLE_CENTER_OFFSET = np.array([0.0, 0.0])

OBSTACLE_RADIUS_X = 0.68
OBSTACLE_RADIUS_Y = 1.20
SLACK_PENALTY = 30000.0

H = 25
DT = 0.3
CONTROL_RATE = 5.0
GOAL_TOLERANCE = 0.10
GOAL_STOP_SPEED = 0.03
TAKEOFF_CLIMB_CMD = 0.5
TAKEOFF_MIN_TIME = 1.5
TAKEOFF_MAX_TIME = 6.0
TAKEOFF_MIN_ALTITUDE_GAIN = 0.20
VEL_FILTER_ALPHA = 0.25
MAX_CMD_SPEED = 0.2
X3_CMD_X_SIGN = -1.0
X3_CMD_Y_SIGN = -1.0
LOG_EVERY_N_TICKS = 3


class RHCControllerNode(Node):
    def __init__(self):
        super().__init__('rhc_controller')
        
        self.cmd_pub = self.create_publisher(Twist, '/X3/gazebo/command/twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/model/X3/odometry', self.odom_callback, 10)
        self.pose_sub = self.create_subscription(TFMessage, '/world/quadcopter/dynamic_pose/info', self.mocap_callback, 10)
        self.pose_sub_all = self.create_subscription(TFMessage, '/world/quadcopter/pose/info', self.mocap_callback, 10)
        
        self.drone_state = np.array([-2.0, 0.0, 0.0, 0.0])
        self.obs_pose = DEFAULT_OBSTACLE_CENTER.copy() + OBSTACLE_CENTER_OFFSET
        self.goal_pose = np.array([GOAL_POSITION[0], GOAL_POSITION[1], 0.0, 0.0])

        self.latest_position = self.drone_state[0:2].copy()
        self.prev_position = self.latest_position.copy()
        self.current_velocity = np.zeros(2)
        self.pose_ready = False
        self.last_pose_time = time.time()
        self.last_pose_debug_time = 0.0
        self.last_frozen_warn_time = 0.0
        self.pose_source = 'none'
        self.has_odom_velocity = False
        self.odom_velocity = np.zeros(2)
        self.pose_freeze_count = 0
        self.latest_altitude = 0.0
        self.takeoff_start_altitude = None

        self.last_drone_time = time.time()
        self.tick_count = 0
        self.last_cmd_vel = np.zeros(2)
        self.current_yaw = 0.0
        self.bypass_phase = 0
        
        self.state = 'TAKEOFF'
        self.start_time = time.time()
        
        self.H = H
        self.dt = DT
        self.control_rate = CONTROL_RATE
        self.setup_mpc()
        self.init_tuning_logs()
        
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        self.get_logger().info("RHC Node Initialized. Commencing Takeoff...")
        self.get_logger().info("State source priority: /model/X3/odometry (primary), TF dynamic/pose info (fallback).")

    def init_tuning_logs(self):
        self.tuning_log_path = os.path.join(os.getcwd(), 'rhc_tuning_log.csv')
        self.tuning_params_path = os.path.join(os.getcwd(), 'rhc_tuning_params.txt')

        with open(self.tuning_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                't', 'state', 'pos_x', 'pos_y', 'vel_x', 'vel_y',
                'cmd_x', 'cmd_y', 'goal_x', 'goal_y', 'obs_x', 'obs_y',
                'dist_goal', 'ellipse_dist', 'bypass_active', 'bypass_phase', 'slack0'
            ])

        with open(self.tuning_params_path, 'w') as f:
            f.write(f'H={self.H}\n')
            f.write(f'DT={self.dt}\n')
            f.write(f'CONTROL_RATE={self.control_rate}\n')
            f.write(f'GOAL_TOLERANCE={GOAL_TOLERANCE}\n')
            f.write(f'TAKEOFF_CLIMB_CMD={TAKEOFF_CLIMB_CMD}\n')
            f.write(f'TAKEOFF_MIN_TIME={TAKEOFF_MIN_TIME}\n')
            f.write(f'TAKEOFF_MAX_TIME={TAKEOFF_MAX_TIME}\n')
            f.write(f'TAKEOFF_MIN_ALTITUDE_GAIN={TAKEOFF_MIN_ALTITUDE_GAIN}\n')
            f.write(f'OBSTACLE_RADIUS_X={OBSTACLE_RADIUS_X}\n')
            f.write(f'OBSTACLE_RADIUS_Y={OBSTACLE_RADIUS_Y}\n')
            f.write(f'SLACK_PENALTY={SLACK_PENALTY}\n')
            f.write(f'MAX_CMD_SPEED={MAX_CMD_SPEED}\n')
            f.write(f'X3_CMD_X_SIGN={X3_CMD_X_SIGN}\n')
            f.write(f'X3_CMD_Y_SIGN={X3_CMD_Y_SIGN}\n')

        self.get_logger().info(f"Tuning logs: {self.tuning_log_path}, {self.tuning_params_path}")

    def append_tuning_log(self, pos_xy, target_goal, dist_to_goal, use_bypass_goal, slack0):
        ellipse_dist = ((pos_xy[0] - self.obs_pose[0])**2) / (OBSTACLE_RADIUS_X**2) + ((pos_xy[1] - self.obs_pose[1])**2) / (OBSTACLE_RADIUS_Y**2)
        with open(self.tuning_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), self.state,
                float(pos_xy[0]), float(pos_xy[1]),
                float(self.current_velocity[0]), float(self.current_velocity[1]),
                float(self.last_cmd_vel[0]), float(self.last_cmd_vel[1]),
                float(target_goal[0]), float(target_goal[1]),
                float(self.obs_pose[0]), float(self.obs_pose[1]),
                float(dist_to_goal), float(ellipse_dist), int(use_bypass_goal), int(self.bypass_phase), float(slack0)
            ])

    def yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def update_pose_state(self, pos_xy, yaw, now, source, vel_xy=None, altitude=None):
        self.latest_position = pos_xy
        self.current_yaw = yaw
        self.last_drone_time = now
        self.last_pose_time = now
        self.pose_source = source
        if altitude is not None:
            self.latest_altitude = float(altitude)

        if vel_xy is not None:
            self.odom_velocity = vel_xy
            self.has_odom_velocity = True

        if not self.pose_ready:
            self.prev_position = pos_xy.copy()
            self.current_velocity = np.zeros(2)
            self.drone_state = np.array([pos_xy[0], pos_xy[1], 0.0, 0.0])
            self.start_time = now
            self.takeoff_start_altitude = self.latest_altitude
            self.pose_ready = True
            self.get_logger().info(
                f"Pose lock acquired from {source} at x={pos_xy[0]:.2f}, y={pos_xy[1]:.2f}, z={self.latest_altitude:.2f}."
            )

        if now - self.last_pose_debug_time > 1.0:
            self.last_pose_debug_time = now
            self.get_logger().info(
                f"Pose[{source}] x={pos_xy[0]:.2f}, y={pos_xy[1]:.2f}, z={self.latest_altitude:.2f}, yaw={yaw:.2f}"
            )

    def select_x3_transform(self, transforms):
        best_transform = None
        best_score = -1e9

        for transform in transforms:
            child = transform.child_frame_id.lower()
            parent = transform.header.frame_id.lower()
            full_name = f"{parent}->{child}"

            score = -1
            if 'x3' in full_name and 'rotor' not in child and 'prop' not in child:
                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                planar_norm = math.hypot(tx, ty)

                score = 0.0
                if child.endswith('/x3') or child == 'x3':
                    score += 8.0
                if 'base_link' in child:
                    score += 6.0
                if 'world' in parent:
                    score += 3.0
                score += min(planar_norm, 5.0)

                if planar_norm < 1e-6:
                    score -= 5.0

            if score > best_score:
                best_score = score
                best_transform = transform

        return best_transform

    def odom_callback(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pos_xy = np.array([px, py])
        yaw = self.yaw_from_quaternion(msg.pose.pose.orientation)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vel_xy = np.array([vx, vy])

        pz = msg.pose.pose.position.z
        self.update_pose_state(pos_xy, yaw, time.time(), 'odom', vel_xy, altitude=pz)

    def setup_mpc(self):
        self.opti = ca.Opti()
        
        self.X = self.opti.variable(4, self.H + 1) 
        self.U = self.opti.variable(2, self.H) 
        self.S = self.opti.variable(self.H)  # Slack variable for soft constraint
        
        self.x_init = self.opti.parameter(4)
        self.x_goal = self.opti.parameter(4)
        self.obs_p = self.opti.parameter(2)
        
        Q = ca.diag([9.0, 4.2, 2.8, 2.8]) / math.sqrt(2.0)
        R = ca.diag([0.70, 0.24]) / math.sqrt(2.0)
        P = ca.diag([50.0, 18.0, 65.0, 65.0]) / math.sqrt(2.0)
        
        cost = 0
        self.opti.subject_to(self.X[:, 0] == self.x_init)
        self.opti.subject_to(self.S >= 0)  # Slack must be non-negative
        
        for k in range(self.H):
            err = self.X[:, k] - self.x_goal
            cost += ca.mtimes([err.T, Q, err]) + ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            cost += SLACK_PENALTY * self.S[k]
            
            x_next = self.X[:, k] + ca.vertcat(self.X[2:4, k] * self.dt, self.U[:, k] * self.dt)
            self.opti.subject_to(self.X[:, k+1] == x_next)
            
            dist_ellipse = ((self.X[0, k+1] - self.obs_p[0])**2) / (OBSTACLE_RADIUS_X**2) + ((self.X[1, k+1] - self.obs_p[1])**2) / (OBSTACLE_RADIUS_Y**2)
            
            self.opti.subject_to(dist_ellipse + self.S[k] >= 2.50)
            
            self.opti.subject_to(self.opti.bounded(-2.0, self.U[0, k], 2.0))
            self.opti.subject_to(self.opti.bounded(-2.0, self.U[1, k], 2.0))
            
        term_err = self.X[:, self.H] - self.x_goal
        cost += ca.mtimes([term_err.T, P, term_err])
        
        self.opti.minimize(cost)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 
                'ipopt.max_iter': 1000, 'ipopt.acceptable_tol': 1e-4}
        self.opti.solver('ipopt', opts)

    def mocap_callback(self, msg):
        x3_tf = self.select_x3_transform(msg.transforms)
        obstacle_tf = None

        for transform in msg.transforms:
            child_name = transform.child_frame_id.lower()
            if 'obstacle' in child_name and obstacle_tf is None:
                obstacle_tf = transform

        if x3_tf is not None and not self.has_odom_velocity:
            tx = x3_tf.transform.translation.x
            ty = x3_tf.transform.translation.y
            tz = x3_tf.transform.translation.z
            pos_xy = np.array([tx, ty])
            yaw = self.yaw_from_quaternion(x3_tf.transform.rotation)
            self.update_pose_state(pos_xy, yaw, time.time(), 'tf', altitude=tz)

        if obstacle_tf is not None:
            self.obs_pose = np.array([obstacle_tf.transform.translation.x, obstacle_tf.transform.translation.y]) + OBSTACLE_CENTER_OFFSET

    def control_loop(self):
        msg = Twist()
        target_goal = self.goal_pose
        slack0 = 0.0

        if not self.pose_ready:
            self.cmd_pub.publish(msg)
            return

        elapsed = time.time() - self.start_time

        pos_xy = self.latest_position.copy()
        if np.linalg.norm(pos_xy - self.prev_position) < 1e-4:
            self.pose_freeze_count += 1
        else:
            self.pose_freeze_count = 0

        if self.state == 'RHC' and self.pose_freeze_count > int(2.0 * self.control_rate):
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            self.cmd_pub.publish(msg)
            self.prev_position = pos_xy
            now = time.time()
            if now - self.last_frozen_warn_time > 2.0:
                self.last_frozen_warn_time = now
                self.get_logger().warn(
                    "Pose appears frozen for >2s; commanding hover for safety. Check ros_gz_bridge pose topic/type mapping."
                )
            return

        if self.has_odom_velocity:
            self.current_velocity = (1.0 - VEL_FILTER_ALPHA) * self.current_velocity + VEL_FILTER_ALPHA * self.odom_velocity
        else:
            raw_velocity = (pos_xy - self.prev_position) * self.control_rate
            self.current_velocity = (1.0 - VEL_FILTER_ALPHA) * self.current_velocity + VEL_FILTER_ALPHA * raw_velocity
        self.drone_state = np.array([pos_xy[0], pos_xy[1], self.current_velocity[0], self.current_velocity[1]])
        
        if self.state == 'TAKEOFF':
            msg.linear.z = TAKEOFF_CLIMB_CMD
            altitude_gain = self.latest_altitude - (self.takeoff_start_altitude if self.takeoff_start_altitude is not None else self.latest_altitude)

            if elapsed >= TAKEOFF_MIN_TIME and altitude_gain >= TAKEOFF_MIN_ALTITUDE_GAIN:
                self.state = 'RHC'
                self.get_logger().info(
                    f"Takeoff complete (dz={altitude_gain:.2f} m). Engaging Receding Horizon Control."
                )
            elif elapsed >= TAKEOFF_MAX_TIME:
                self.get_logger().warn(
                    f"Takeoff not confirmed after {TAKEOFF_MAX_TIME:.1f}s (dz={altitude_gain:.2f} m). Holding climb command; check twist bridge."
                )
                
        elif self.state == 'RHC':
            dist_to_goal = np.linalg.norm(pos_xy - GOAL_POSITION)
            speed_now = np.linalg.norm(self.current_velocity)
            if dist_to_goal < GOAL_TOLERANCE and speed_now < GOAL_STOP_SPEED:
                self.state = 'HOVER'
                self.get_logger().info(f"Goal Reached! Hovering. dist={dist_to_goal:.3f}, speed={speed_now:.3f}")
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                self.last_cmd_vel = np.zeros(2)
            else:
                target_goal = self.goal_pose

                self.opti.set_value(self.x_init, self.drone_state)
                self.opti.set_value(self.x_goal, target_goal)
                self.opti.set_value(self.obs_p, self.obs_pose)
                
                try:
                    self.opti.set_initial(self.U, np.ones((2, self.H)) * 0.1)
                    self.opti.set_initial(self.S, np.zeros(self.H))
                    
                    for k in range(self.H + 1):
                        self.opti.set_initial(self.X[:, k], self.drone_state)
                    
                    sol = self.opti.solve()
                    u_opt = sol.value(self.U)
                    slack0 = float(sol.value(self.S[0]))
                    
                    ax_cmd = u_opt[0, 0] if self.H > 1 else u_opt[0]
                    ay_cmd = u_opt[1, 0] if self.H > 1 else u_opt[1]

                    v_cmd = self.current_velocity + np.array([ax_cmd, ay_cmd]) * self.dt
                    alpha = 0.15
                    cmd_vel = self.current_velocity + alpha * (v_cmd - self.current_velocity)

                    cmd_speed = np.linalg.norm(cmd_vel)
                    if cmd_speed > MAX_CMD_SPEED:
                        cmd_vel = cmd_vel * (MAX_CMD_SPEED / cmd_speed)

                    cy = math.cos(self.current_yaw)
                    sy = math.sin(self.current_yaw)
                    body_vx = cy * cmd_vel[0] + sy * cmd_vel[1]
                    body_vy = -sy * cmd_vel[0] + cy * cmd_vel[1]

                    msg.linear.x = float(-X3_CMD_X_SIGN * body_vx)
                    msg.linear.y = float(-X3_CMD_Y_SIGN * body_vy)
                    msg.linear.z = 0.0
                    self.last_cmd_vel = np.array([msg.linear.x, msg.linear.y])

                    self.opti.set_initial(sol.value_variables())
                        
                except RuntimeError:
                    self.get_logger().warn("MPC solver failed this tick. Holding velocity.")
                    msg.linear.x = 0.0
                    msg.linear.y = 0.0
                    self.last_cmd_vel = np.zeros(2)
                
        elif self.state == 'HOVER':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0

        self.cmd_pub.publish(msg)
        self.prev_position = pos_xy
        self.tick_count += 1

        if self.tick_count % LOG_EVERY_N_TICKS == 0:
            self.append_tuning_log(pos_xy, target_goal, float(np.linalg.norm(pos_xy - GOAL_POSITION)), False, slack0)

def main(args=None):
    rclpy.init(args=args)
    node = RHCControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()