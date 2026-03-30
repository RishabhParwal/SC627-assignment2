import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import time

class X3FeederNode(Node):
    def __init__(self):
        super().__init__('open_loop_feeder')
        
        # Publisher to the Gazebo Velocity Controller
        self.cmd_pub = self.create_publisher(Twist, '/X3/gazebo/command/twist', 10)
        
        # Load the CSV
        self.trajectory = self.load_trajectory('optimal_trajectory.csv')
        self.n_points = len(self.trajectory)
        self.idx = 0
        
        # Mission State
        self.state = 'TAKEOFF'
        self.start_time = time.time()
        
        # Run at 50Hz
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Node started. Initiating blind takeoff...")

    def load_trajectory(self, filename):
        traj = []
        try:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    traj.append({
                        't': float(row['t']), 
                        'vx': float(row['vx']), 'vy': float(row['vy']), 'vz': float(row['vz'])
                    })
            self.get_logger().info(f"Loaded {len(traj)} waypoints.")
            return traj
        except FileNotFoundError:
            self.get_logger().error(f"Cannot find {filename}")
            exit(1)

    def control_loop(self):
        msg = Twist()
        now = time.time()
        elapsed = now - self.start_time
        
        if self.state == 'TAKEOFF':
            if elapsed < 3.0:
                # Climb straight up at 0.5 m/s
                msg.linear.z = 0.5 
            else:
                self.state = 'TRAJECTORY'
                self.start_time = time.time()  # Reset clock for the trajectory
                self.get_logger().info("Executing Open-Loop Trajectory blindly based on time!")
                
        elif self.state == 'TRAJECTORY':
            if self.idx < self.n_points:
                target_t = self.trajectory[self.idx]['t']
                
                # Advance the index if time has passed the current waypoint
                if elapsed >= target_t:
                    self.idx += 1
                
                # Feed the velocity blindly
                active_pt = self.trajectory[min(self.idx, self.n_points - 1)]
                msg.linear.x = active_pt['vx']
                msg.linear.y = active_pt['vy']
                
                # Keep altitude locked to 0 velocity (hover level) during the run
                msg.linear.z = 0.0 
            else:
                self.state = 'HOVER'
                self.get_logger().info("Trajectory complete. Hovering.")
                
        elif self.state == 'HOVER':
            # Zero out velocities to stop
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0

        # Publish the command to Gazebo
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = X3FeederNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()