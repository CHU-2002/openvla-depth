# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import (
    AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor,AutoTokenizer
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor, PrismaticImageProcessor
import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import threading
from peft import PeftModel
from ur5_controller import RealTimeUR5ControllerUsingMoveL, AsyncGripperController, CameraManager, next_dataset_directory

# -----------------------------
# 1. 加载 OpenVLA 模型与 Processor
# -----------------------------
print("[INFO] Loading OpenVLA model and processor...")
local_model_path = "/mnt/disk4/yitong/changedcafattokcat/31f090d05236101ebfc381b61c674dd4746d4ce0+ur5_robo_dataset+b8+lr-0.0001+lora-r32+dropout-0.0--image_aug--25000_chkpt"
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    attn_implementation="flash_attention_2",  # 可选项，需要 flash_attn 支持
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda:0")

# model_path = "openvla/openvla-7b"

# # Load config & base model
# config = OpenVLAConfig.from_pretrained(model_path, trust_remote_code=True)

# base = OpenVLAForActionPrediction.from_pretrained(model_path, config=config, trust_remote_code=True)

# image_processor = PrismaticImageProcessor.from_pretrained(model_path)

# # 加载 tokenizer（照常）
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # 用你本地两个组件构造 processor
# processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

# # Load LoRA adapter & cast model to bfloat16 + cuda  ✅ MODIFIED
# adapter_dir = "./adaptor"
# vla = PeftModel.from_pretrained(base, adapter_dir)
# # vla.eval()



MAX_RES = 224

new_states = {
"action": {
      "mean": [
        0.00034084549406543374,
        -0.00030492557561956346,
        -0.0008138284320011735,
        0.016729077324271202,
        -1.8060404727293644e-07,
        2.659022015905066e-07,
        0.5941628217697144
      ],
      "std": [
        0.0036212557461112738,
        0.005531256087124348,
        0.0048943376168608665,
        4.131305694580078,
        0.00022504849766846746,
        0.0002433750923955813,
        0.49105530977249146
      ],
      "max": [
        0.01611638069152832,
        0.019588030874729156,
        0.018839538097381592,
        6.283182144165039,
        0.0011729226680472493,
        0.001676865853369236,
        1.0
      ],
      "min": [
        -0.019087255001068115,
        -0.01944923773407936,
        -0.01672576367855072,
        -6.2831830978393555,
        -0.0013649603351950645,
        -0.001678195665590465,
        0.0
      ],
      "q01": [
        -0.009785709381103515,
        -0.01367771863937378,
        -0.011039825081825256,
        -6.283165626525879,
        -0.0005680103786289692,
        -0.0005832317052409053,
        0.0
      ],
      "q99": [
        0.0100857949256897,
        0.01377965800464154,
        0.01329185485839849,
        6.283167839050293,
        0.0005469654710032047,
        0.0005947400396689773,
        1.0
      ],
      "mask": [
        True,
        True,
        True,
        True,
        True,
        True,
        False
      ]
    },
    "proprio": {
      "mean": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "std": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "max": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "min": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "q01": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "q99": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    "num_transitions": 9765,
    "num_trajectories": 97
}

vla.norm_stats["ur5"] = new_states

# -----------------------------
# 2. 初始化控制接口
# -----------------------------
# 修改为实际的 UR5 机器人控制器 IP 地址和摄像头序列号
UR5_IP = "192.168.1.60"
CAMERA_MAP= {
    "side_1": "218622277783",#405
    # "side_2": "819612070593",#435
    # "wrist_1": "130322272869" #D405
}
SAVE_PATH = "data"  # 数据保存目录（CameraManager 内部可保存采集的数据）
SAVE_PATH = next_dataset_directory(SAVE_PATH)

print("[INFO] Initializing robot control interfaces...")
robot_controller = RealTimeUR5ControllerUsingMoveL(UR5_IP)
gripper_controller = AsyncGripperController(UR5_IP)
camera_manager = CameraManager(camera_map=CAMERA_MAP, save_path=SAVE_PATH, UR5_controller=robot_controller, gripper_controller=gripper_controller)
camera_manager.init_cameras()
camera_thread = threading.Thread(target=camera_manager.run, daemon=True)
camera_thread.start()

current_pose = [-0.320, -0.060, 0.320, 0, np.pi, 0]
robot_controller.send_pose_to_robot(current_pose)
robot_controller.start_realtime_control(frequency=5)
gripper_controller.control_gripper(close=False, force=100, speed=30)
time.sleep(5)
# -----------------------------
# 3. 定义控制循环
# -----------------------------
def main():
    # 定义任务指令，此处以 "put the bottle on the carton" 为例
    instruction = "put the white cube into the bule box"
    # 构造 prompt 模板；请确保 prompt 格式与模型训练时一致

    print("[INFO] Starting main control loop...")
    try:
        while True:
            pose, _= robot_controller.get_robot_pose_and_joints()
            gripper = gripper_controller.get_gripper_state()
            if gripper == True:
                gs = [1.0]
            else:
                gs = [0.0]
            n = [0.0]
            proprio = pose + n + gs

            prompt = f"What action should the robot take to {instruction}, given robot proprioception {proprio}?"
            print(prompt)
            # 通过 CameraManager 获取最新的 "wrist" 摄像头图像
            frame_data = camera_manager.get_latest_frame("side_1")
            if frame_data is None:
                # 尚未采集到图像时等待
                time.sleep(0.1)
                continue

            # 从采集数据中提取 color 图像（OpenCV 格式：BGR）
            image_cv = frame_data["color"]
            # print(image_cv)
            # 转换为 PIL Image（RGB 格式）
            image_cv = image_cv[:, :, ::-1]
            image_cv = cv2.resize(image_cv, (MAX_RES, MAX_RES))
            image_pil = Image.fromarray(image_cv)
            depth_cv = frame_data["depth"]
            depth_cv = cv2.resize(image_cv, (MAX_RES, MAX_RES))
            depth_pil = Image.fromarray(depth_cv)
            # image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            # 如果需要，可以在此处对图像进行左右翻转，使其与模型训练时预处理一致
            # image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)

            # -----------------------------
            # 3.1 利用 OpenVLA 模型预测动作
            # -----------------------------
            # 将 prompt 与图像输入 processor，注意返回的是 torch.Tensor（默认为 batch_size=1）
            inputs = processor(instruction, image_pil, depth_images=depth_pil, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)
            # 调用 predict_action 生成动作，使用 unnorm_key="bridge_orig" 反归一化
            action = vla.predict_action(**inputs, unnorm_key="ur5", do_sample=False)
            # 这里假设 action 是一个列表或数组，包含 7 个数值：
            # 前 6 个控制机器人位姿（位置和姿态）的增量，第 7 个用于吸盘控制

            if isinstance(action, torch.Tensor):
                action = action.cpu().detach().numpy().flatten()
            else:
                action = action.flatten()

            print("[INFO] Predicted action:", action)

            # -----------------------------
            # 3.2 执行预测的动作
            # -----------------------------
            # 获取当前机器人位姿（假定返回 6 个数值：[x, y, z, Rx, Ry, Rz]）

            # 计算新的目标位姿（简单地将当前位姿与动作增量相加）
            for i in range(3):
                current_pose[i] = current_pose[i] + action[i]
            print("[INFO] Sending new pose to robot:", current_pose)
            robot_controller.send_pose_to_robot(current_pose)

            # # 根据动作第 7 个数值控制吸盘：例如，当该值大于 0.5 时闭合，否则打开
            if action[6] > 0.1:
                gripper_controller.control_gripper(close=True, force=300, speed=100)
                print("[INFO] Closing gripper.")
            else:
                gripper_controller.control_gripper(close=False, force=300, speed=100)
                print("[INFO] Opening gripper.")

            # 可根据需要调整循环延时
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected, exiting control loop.")
    finally:
        # 关闭各控制接口
        camera_manager.stop()
        if camera_thread.is_alive():
            camera_thread.join()
        robot_controller.close()
        gripper_controller.disconnect()
        print("[INFO] Program finished.")

if __name__ == "__main__":
    main()
