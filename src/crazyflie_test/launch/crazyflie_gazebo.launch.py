import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("crazyflie_test")
    world_file = os.path.join(pkg_share, "worlds", "quadcopter.sdf")

    return LaunchDescription([
        SetEnvironmentVariable(
            name="IGN_IP",
            value="127.0.0.1",
        ),
        SetEnvironmentVariable(
            name="GZ_IP",
            value="127.0.0.1",
        ),
        SetEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=os.path.join(pkg_share, "worlds") + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        ),
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=os.path.join(pkg_share, "worlds") + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
            ),
            launch_arguments={"gz_args": ["-r ", world_file]}.items(),
        ),
    ])