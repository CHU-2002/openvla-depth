import pyrealsense2 as rs
import cv2
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time
import asyncio
import numpy as np
import os
import threading
from .vacuum_gripper import VacuumGripper

# Example camera serial dictionary
REALSENSE_CAMERAS = {
    "side_1": "218622277783",#405
    "side_2": "819612070593",#435
    "wrist_1": "130322272869" #D405
    # "side_2": "239722072823"
}


class RealTimeUR5Controller:
    """
    This sample class demonstrates how to update the UR5 end-effector pose in real-time
    (e.g., at 5Hz) using servoL, and also how to control the gripper or retrieve the
    robot's current pose/joint positions.

    Example usage:
        controller = RealTimeUR5Controller("192.168.0.100")
        controller.start_realtime_control(frequency=5.0)

        # In an external loop, keep updating the pose
        for step in range(50):
            new_pose = [0.3, -0.2 + 0.001*step, 0.2, 0, 0, 0]
            controller.send_pose_to_robot(new_pose, speed=1.0, acceleration=0.8)
            time.sleep(0.2)  # corresponding to 5Hz

        controller.stop_realtime_control()
        # Optionally control the gripper
        controller.control_gripper(close=True)
        # Also query current pose
        pose, joints = controller.get_robot_pose_and_joints()
        controller.close()
    """

    def __init__(self, ip_address: str, speed=0.1, acceleration=0.1):
        """
        :param ip_address: UR5 robot controller IP address
        """
        if ip_address:
            self.ip_address = ip_address
        else :
            self.ip_address = "192.168.1.60"

        # RTDE control & receive interfaces
        self.rtde_control = RTDEControlInterface(ip_address)
        self.rtde_receive = RTDEReceiveInterface(ip_address)

        self._running = False
        self._control_thread = None

        # Default: get the current pose from the robot
        self._target_pose = self.rtde_receive.getActualTCPPose()
        # Default speed & acceleration for moveL
        self._speed = speed
        self._acceleration = acceleration

        print(f"[INFO] Connected to UR5 at {ip_address}")

    def start_realtime_control(self, frequency=100.0):
        """
        Start a background thread that calls servoL() at the specified frequency (Hz),
        so the robot EEF follows self._target_pose.
        """
        if self._running:
            print("[WARNING] Real-time control is already running.")
            return

        self._running = True
        self._control_thread = threading.Thread(
            target=self._control_loop,
            args=(frequency,),
            daemon=True
        )
        self._control_thread.start()
        print(f"[INFO] Real-time control started at {frequency} Hz.")

    def _control_loop(self, frequency):
        """
        Background thread: call servoL() every 1/frequency seconds, moving toward self._target_pose.
        """
        period = 1.0 / frequency
        while self._running:
            self.rtde_control.servoL(
                self._target_pose,
                self._speed,
                self._acceleration,
                period,
                0.1,
                300
            )
            time.sleep(period)

        # If you want to stop the motion gracefully, call stopL()
        self.rtde_control.stopL(2.0)
        print("[INFO] Real-time control loop stopped.")

    def stop_realtime_control(self):
        """
        Stop the background real-time control thread.
        """
        if not self._running:
            return
        self._running = False
        if self._control_thread is not None:
            self._control_thread.join()
        print("[INFO] Real-time control stopped.")

    def send_pose_to_robot(self, pose, speed=0.1, acceleration=0.1):
        """
        External call to update self._target_pose.
        The background thread will keep calling servoL() to move toward this new pose.

        :param pose: [x, y, z, Rx, Ry, Rz] (Rx, Ry, Rz as axis-angle)
        :param speed: (m/s)
        :param acceleration: (m/s^2)
        """
        self._target_pose = pose
        self._speed = speed
        self._acceleration = acceleration

    def get_robot_pose_and_joints(self):
        """
        Get the current EEF pose (x, y, z, Rx, Ry, Rz) and 6 joint angles.

        Returns: (task_space_pose, joint_space_positions)
        """
        try:
            # Current TCPPose
            task_space_pose = self.rtde_receive.getActualTCPPose()
            # Example: negate Rx, Ry, Rz
            task_space_pose = task_space_pose[:3] + [-val for val in task_space_pose[3:]]

            # Current joint angles
            joint_space_positions = self.rtde_receive.getActualQ()
            return task_space_pose, joint_space_positions
        except Exception as e:
            print("Error while retrieving robot data:", e)
            return None, None

    def close(self):
        """
        Stop the background control thread and release RTDE resources.
        """
        self.stop_realtime_control()
        self.rtde_control.stopScript()
        print("[INFO] Robot controller closed.")




# def main():
#     UR5_IP = "192.168.1.60"
#     SAVE_PATH = "data"

#     # 1. Instantiate UR5 and gripper controllers
#     ur5_controller = RealTimeUR5Controller(UR5_IP)
#     gripper_controller = AsyncGripperController(UR5_IP)

#     # 2. Instantiate CameraManager
#     camera_manager = CameraManager(
#         camera_map=REALSENSE_CAMERAS,
#         save_path=SAVE_PATH,
#         UR5_controller=ur5_controller,
#         gripper_controller=gripper_controller
#     )

#     # 3. Initialize cameras
#     camera_manager.init_cameras()
#     # 4. Prepare a thread to run camera_manager.run() for collecting and storing data
#     camera_thread = threading.Thread(target=camera_manager.run, daemon=True)
#     # camera_thread.start()
#     # 5. Start a simple SpaceMouse control
#     spacemouse = SpaceMouseExpert()  # You need to implement or use a third-party library
#     camera_running = False
#     zero_count = 0
#     ur5_controller.start_realtime_control(frequency=5.0)
#     # Open the gripper initially
#     gripper_controller.control_gripper(close=False, force=100, speed=30)

#     # Set an initial robot pose
#     current_pose = [-0.470, -0.180, 0.450, np.pi, 0, 0]
#     print("current_pose: ", current_pose)
#     print("type",type(current_pose))
    
#     ur5_controller.send_pose_to_robot(current_pose)

#     print("[INFO] Start main loop for SpaceMouse + Robot ...")

#     try:
#         while True:
#             # Get SpaceMouse movement (action) and buttons
#             action, buttons = spacemouse.get_action()
#             print(f"Spacemouse action: {action}, buttons: {buttons}")

#             # Update current_pose
#             for i in range(6):
#                 current_pose[i] += action[i] * 0.05

#             # Send new EEF pose
#             ur5_controller.send_pose_to_robot(current_pose, speed=0.8, acceleration=0.8)

#             # Use button[0] to control gripper
#             close_gripper = bool(buttons[0])
#             gripper_controller.control_gripper(close=close_gripper, force=100, speed=100)

#             # Check if there's any movement/button input
#             if any(abs(a) > 1e-5 for a in action) or any(buttons):
#                 zero_count = 0
#                 if not camera_running:
#                     # Start camera thread
#                     camera_thread.start()
#                     camera_running = True
#                     print("Camera thread started.")
#             # else:
#             #     zero_count += 1
#             #     # If no input for 5 cycles, exit
#             #     if zero_count >= 5:
#             #         if camera_running:
#             #             print("Stopping main loop since no input for 5 cycles.")
#             #         break

#             time.sleep(0.2)

#     except KeyboardInterrupt:
#         print("[INFO] KeyboardInterrupt: stopping main loop.")
#     finally:
#         # 6. Stop CameraManager
#         camera_manager.stop()
#         if camera_thread.is_alive():
#             camera_thread.join()

#         # 7. Disconnect gripper and close UR5 RTDE
#         gripper_controller.disconnect()
#         ur5_controller.close()

#         print("[INFO] Main program finished.")
#     # time.sleep(10)
#     # camera_manager.stop()
#     # # if camera_thread.is_alive():
#     # #     camera_thread.join()


# if __name__ == "__main__":
#     main()
