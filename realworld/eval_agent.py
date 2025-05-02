"""
Evaluation Agent.
"""

import cv2
import time
import numpy as np

from get_ip import get_ip
from device.camera import CameraD400
from device.robot import FlexivRobot, FlexivGripper

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from utils.transformation import xyz_rot_transform

class Agent:
    """
    Evaluation agent with Flexiv arm, Dahuan gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        cam_ids = []
    ): 
        self.cam_ids = cam_ids
        print("Init robot, gripper, and camera.")
        robot_ip, pc_ip = get_ip()
        print(robot_ip, pc_ip)
        self.robot = FlexivRobot(robot_ip_address = robot_ip, pc_ip_address = pc_ip)
        self.gripper = FlexivGripper(self.robot)
        self.reset()
        self.cameras = [CameraD400(serial = cam_id) for cam_id in self.cam_ids]
        print("Initialization Finished.")
    
    def reset(self):
        self.robot.send_tcp_pose(self.ready_pose)
        time.sleep(1.5)
        self.gripper.move(self.gripper.max_width)
        time.sleep(0.5)

    @property
    def ready_pose(self):
        # return np.array([0.6,0,0.2,0,0.5**0.5,0.5**0.5,0])
        # return np.array([0.6,0,0.2,1.0,0,0,0])
        # return np.array([0.5, 0.0, 0.17, 0.0, 0.0, 1.0, 0.0])
        return np.array([0.6, 0, 0.2, 0, -0.5**0.5, 0.5**0.5, 0])

    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0])

    def get_observation(self):
        # colors, depths = self.camera.get_data()
        colors_dict = {}
        depths_dict = {}
        for i, camera in zip(self.cam_ids, self.cameras):
            colors, depths = camera.get_data()
            colors_dict[i] = cv2.cvtColor(colors.copy(), cv2.COLOR_BGR2RGB)
            depths_dict[i] = depths.copy() / 1000.
        return colors_dict, depths_dict
    
    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention = None, blocking = False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )
        self.robot.send_tcp_pose(tcp_pose)
        if blocking:
            time.sleep(0.1)
    
    def set_gripper_width(self, width, blocking = False):
        self.gripper.move(width)
        if blocking:
            time.sleep(0.5)
    
    def stop(self):
        self.robot.stop()
    
if __name__ == "__main__":
    agent = Agent()
    agent.robot.send_tcp_pose(np.array([0.6, 0, 0.2, 0, -0.5**0.5, 0.5**0.5, 0]))
    time.sleep(1.5)
    agent.stop()
