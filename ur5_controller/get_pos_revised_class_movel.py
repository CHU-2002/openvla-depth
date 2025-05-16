import pyrealsense2 as rs
import cv2
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time
import asyncio
import numpy as np
import os
import threading
import re
from .vacuum_gripper import VacuumGripper
# Example camera serial dictionary
REALSENSE_CAMERAS = {

    "side_1": "218622277783",#405
    # "side_2": "819612070593",#435
    # "wrist_1": "130322272869" #D405
    # "side_2": "239722072823"
}



DEBUG = False
class CameraManager:
    """
    1) 初始化多路 RealSense 摄像头；
    2) 持续采集 color 与 depth 数据；
    3) 通过传入的 UR5 与吸盘控制器获取机器人状态，并保存图像及数据。
    """
    def __init__(self, camera_map: dict, save_path, UR5_controller=None, gripper_controller=None):
        self.camera_map = camera_map
        self.save_path = save_path
        self.pipelines = {}   # camera_name -> rs.pipeline() 对象
        self.index = 0        # 保存数据的计数器
        self.latest_frames = {}  # {camera_name: {"color": ..., "depth": ...}, ...}
        self.ur5_controller = UR5_controller
        self.gripper_controller = gripper_controller

    def init_cameras(self):
        os.makedirs(self.save_path, exist_ok=True)
        for name, serial in self.camera_map.items():
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pipeline.start(config)
                self.pipelines[name] = pipeline
                print(f"Camera [{name}] with serial={serial} started successfully.")
            except Exception as e:
                print(f"Error starting camera [{name}] with serial={serial}: {e}")
                self.pipelines[name] = None

    def run(self):
        try:
            while True:
                all_imgs = []
                for name, pipeline in self.pipelines.items():
                    if pipeline is None:
                        continue
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    self.latest_frames[name] = {"color": color_image, "depth": depth_image}
                    all_imgs.append(color_image)
                    all_imgs.append(depth_image)
                    cv2.imshow(f"Camera {name}", color_image)
                if self.ur5_controller:
                    pos, joint = self.ur5_controller.get_robot_pose_and_joints()
                else:
                    pos = [0.3, -0.2, 0.2, 0, 0, 0]
                    joint = [0.0] * 6
                if self.gripper_controller:
                    gripper_state = self.gripper_controller.get_gripper_state()
                else:
                    gripper_state = 0
                if len(all_imgs) >= 2:
                    self._save_data(
                        index=self.index,
                        joint=joint,
                        pos=pos,
                        imgs=all_imgs,
                        gripper=gripper_state
                    )
                    print(f"[INFO] Saved data at index={self.index}")
                    self.index += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Exiting camera loop.")
                    break
                time.sleep(0.05)
        except Exception as e:
            print(f"[ERROR] Exception in camera streaming loop: {e}")
        finally:
            self.stop()

    def get_latest_frame(self, camera_name: str = None):
        if camera_name is None:
            if not self.latest_frames:
                return None
            return next(iter(self.latest_frames.values()))
        else:
            return self.latest_frames.get(camera_name, None)

    def _save_data(self, index, joint, pos, imgs, gripper):
        primary_color_img = primary_depth_img = None
        if len(imgs) >= 2:
            primary_color_img = imgs[0]
            primary_depth_img = imgs[1]
        data = {
            'joint': np.array(joint, dtype=np.float32),
            'pose':  np.array(pos, dtype=np.float32),
            'primary_image': primary_color_img,
            'primary_depth_image': primary_depth_img,
            'gripper': gripper
        }
        if primary_color_img is not None:
            primary_color_path = os.path.join(self.save_path, f"primary_{index}.jpg")
            cv2.imwrite(primary_color_path, primary_color_img)
        np.save(os.path.join(self.save_path, f"targ{index}.npy"), data)

    def stop(self):
        for name, pipeline in self.pipelines.items():
            if pipeline is not None:
                pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] CameraManager stopped.")


class RealTimeUR5ControllerUsingMoveL:
    """
    A simplified example demonstrating how to continuously update
    the UR5 end-effector pose using moveL (with async=True),
    for those cases where Polyscope is too old to support servoL().
    
    WARNING:
      - This approach calls moveL() in a loop at some frequency
        and updates the final target each time.
      - The motion may be less smooth than servoL() and might
        get interrupted frequently if the target changes rapidly.
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

    def start_realtime_control(self, frequency=5.0):
        """
        Start a background thread at 'frequency' Hz. Each loop calls moveL()
        with async=True toward self._target_pose.

        NOTE: This can lead to less smooth movement if _target_pose changes often.
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
        print(f"[INFO] Real-time control started at {frequency} Hz (moveL).")

    def _control_loop(self, frequency):
        """
        Background thread: every 1/frequency seconds, call moveL(..., async=True)
        to move toward the latest _target_pose.
        """
        period = 1.0 / frequency
        while self._running:
            # Fire off an asynchronous linear move to the current target pose
            # If the user changes self._target_pose in between calls, it might
            # interrupt the previous move.
            try:
                # print("pose",self._target_pose)
                self.rtde_control.moveL(
                    self._target_pose,
                    speed=self._speed,
                    acceleration=self._acceleration
                )
            except Exception as e:
                print("[ERROR] moveL failed:", e)
                # You might want to break or continue based on your needs
                break

            # Sleep to maintain the desired update rate
            # time.sleep(period)

        print("[INFO] Real-time control loop stopped.")

    def stop_realtime_control(self):
        """
        Stop the background control thread.
        """
        if not self._running:
            return
        self._running = False
        if self._control_thread is not None:
            self._control_thread.join()
        print("[INFO] Real-time control stopped.")

    def send_pose_to_robot(self, pose):
        """
        Update the internal target pose; the background thread
        will repeatedly call moveL(...) to move toward this new pose.

        :param pose: [x, y, z, Rx, Ry, Rz] in UR base coordinates
                     (Rx, Ry, Rz in axis-angle, typically).
        :param speed: max linear speed (m/s)
        :param acceleration: max linear acceleration (m/s^2)
        """
        self._target_pose = pose

    def get_robot_pose_and_joints(self):
        """
        Get the current EEF pose (x, y, z, Rx, Ry, Rz) and 6 joint angles.

        Returns: (task_space_pose, joint_positions)
        """
        try:
            # Current TCPPose
            task_space_pose = self.rtde_receive.getActualTCPPose()

            # Example: negate Rx, Ry, Rz if you want that convention;
            # otherwise just return as-is:
            task_space_pose = task_space_pose[:3] + [-val for val in task_space_pose[3:]]

            # Current joint angles
            joint_positions = self.rtde_receive.getActualQ()
            return task_space_pose, joint_positions
        except Exception as e:
            print("Error retrieving robot data:", e)
            return None, None

    def close(self):
        """
        Stop the background control thread and release RTDE resources.
        """
        self.stop_realtime_control()
        self.rtde_control.stopScript()
        print("[INFO] Robot controller closed.")

class AsyncGripperController:
    """
    This class demonstrates how to establish a single VacuumGripper connection and
    reuse it throughout the class lifecycle for repeated open/close operations
    without reconnecting each time.

    Example usage:
        controller = AsyncGripperController("192.168.0.100")
        # Open the gripper
        controller.control_gripper(close=False, force=80, speed=25)
        # Close the gripper
        controller.control_gripper(close=True, force=100, speed=30)
        # Finally disconnect
        controller.disconnect()
    """

    def __init__(self, ip_address: str):
        self.ip_address = ip_address
        self.loop = None
        self.gripper = None
        self._connected = False
        self._gripper_state = False  # False for open, True for closed

    def _ensure_event_loop(self):
        """
        Ensure self.loop exists and is usable. Create a new loop if none is found.
        """
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

    async def _init_gripper(self):
        """
        Asynchronously initialize and activate the vacuum gripper, only once.
        """
        if self.gripper is None:
            self.gripper = VacuumGripper(self.ip_address)

        if not self._connected:
            await self.gripper.connect()
            await self.gripper.activate()
            self._connected = True

    def get_gripper_state(self):
        return self._gripper_state

    def _run_async_task(self, coro):
        """
        Run the coroutine in an existing (or newly created) event loop.
        If the loop is already running, use ensure_future; otherwise run_until_complete.
        """
        self._ensure_event_loop()

        if self.loop.is_running():
            asyncio.ensure_future(coro, loop=self.loop)
        else:
            self.loop.run_until_complete(coro)

    def control_gripper(self, close=True, force=100, speed=30):
        """
        Control gripper open/close:
         - close=True for closing
         - close=False for opening
         - force & speed for gripper settings
        """
        async def _control():
            await self._init_gripper()
            if close:
                self._gripper_state = True
                await self.gripper.close_gripper(force=force, speed=speed)
                print("Gripper closed.")
            else:
                self._gripper_state = False
                await self.gripper.open_gripper(force=force, speed=speed)
                print("Gripper opened.")

        self._run_async_task(_control())

    def disconnect(self):
        """
        Disconnect from the gripper. The next control_gripper() call will reconnect.
        """
        async def _disconnect():
            if self.gripper and self._connected:
                await self.gripper.disconnect()
                self._connected = False
                self.gripper = None

        self._run_async_task(_disconnect())

    def __del__(self):
        """
        Destructor: attempt to disconnect on garbage collection.
        It's recommended to explicitly call disconnect() rather than rely on __del__.
        """
        try:
            self.disconnect()
        except:
            pass

def next_dataset_directory(save_path):
    # 列出所有以 "task" 开头的目录
    existing_tasks = [d for d in os.listdir(save_path) if d.startswith("task")]
    
    if not existing_tasks:
        next_task_number = 1
    else:
        # 提取任务目录中的数字部分并按数字排序
        task_numbers = []
        for task in existing_tasks:
            match = re.match(r"task(\d+)", task)
            if match:
                task_numbers.append(int(match.group(1)))
        
        # 获取下一个序号
        next_task_number = max(task_numbers) + 1
    
    # 返回下一个数据集应该存储的目录路径
    next_dataset_dir = os.path.join(save_path, f"task{next_task_number}")
    return next_dataset_dir



def main():
    UR5_IP = "192.168.1.60"
    SAVE_PATH = "data"
    
    camera_manager = CameraManager(
        camera_map=REALSENSE_CAMERAS,
        save_path=SAVE_PATH,
    )

    camera_manager.init_cameras()
    camera_thread = threading.Thread(target=camera_manager.run, daemon=True)
    camera_thread.start()

    try:
        while True:
            color_image = camera_manager.get_latest_frame("side_1")
            if color_image:
                print("Camera View", color_image["color"])
                
            # 等待 1ms，按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Stopping Camera Manager.")

    finally:
        camera_manager.stop()
        camera_thread.join()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
