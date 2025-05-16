# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import (
    AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor,AutoTokenizer
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor, PrismaticImageProcessor
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import torch
import cv2
import time
from prismatic.vla.action_tokenizer import ActionTokenizer
from peft import PeftModel


def print_depth_statistics(dep_primary_image):
    """
    计算并打印深度图像的统计信息，包括最小值、最大值、均值、标准差，
    以及基于 50 个区间的直方图统计。
    """
    # 计算统计数据
    depth_min = np.min(dep_primary_image)
    depth_max = np.max(dep_primary_image)
    depth_mean = np.mean(dep_primary_image)
    depth_std = np.std(dep_primary_image)
    
    print("Depth 图像统计信息:")
    print(f"  数据范围: {depth_min} ~ {depth_max}")
    print(f"  均值: {depth_mean:.2f}")
    print(f"  标准差: {depth_std:.2f}")
    
    # 计算直方图，先将数据摊平成一维数组
    hist, bin_edges = np.histogram(dep_primary_image.flatten(), bins=50)
    print("直方图统计:")
    print("  Bin 边界:", bin_edges)
    print("  每个区间内像素数量:", hist)
print("[INFO] Loading OpenVLA model and processor...")





local_model_path = "/mnt/second_drive/home/Openvla_log/poseadd/openvla-7b+ur5_robo_dataset+b16+lr-0.0001+lora-r32+dropout-0.0--image_aug"
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    attn_implementation="flash_attention_2",  # 可选项，需要 flash_attn 支持
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda:0")
# print(vla)
# config = OpenVLAConfig.from_pretrained(model_path, trust_remote_code=True)

# base = OpenVLAForActionPrediction.from_pretrained(model_path, config=config, trust_remote_code=True).to("cuda0")  # 如果在 GPU 上推理
# base.eval()

# image_processor = PrismaticImageProcessor.from_pretrained(model_path)

# # 加载 tokenizer（照常）
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # 用你本地两个组件构造 processor
# processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
# action_tokenizer = ActionTokenizer(processor.tokenizer)

# adapter_dir = "./adaptor"

# # 将基座模型包装成带 LoRA 的模型
# vla = PeftModel.from_pretrained(
#     base,     # 第一步加载好的 base_model
#     adapter_dir,    # LoRA adapter 文件夹
# )
# vla.eval()
# vla.to("cuda0")  # 确保在相同设备上


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

try:
    vla.print_trainable_parameters()  
except AttributeError:
    print("若出现报错或提示没有 Peft 相关方法，也说明它已不是一个 PEFT 模型了。")

#   - 另一种方法是直接数一下 requires_grad=True 的参数：
trainable_params = [p for p in vla.parameters() if p.requires_grad]
print(len(trainable_params), "trainable params after merge")

for i in range(150):
    data = np.load(f"task1/targ{i}.npy", allow_pickle=True).item()  # 这里要用 .item() 取出字典
    # 获取 primary_image 和 pose
    print(f"now is prediction on targ{i}.npy")
    primary_image = data.get('primary_image', None)
    primary_image = primary_image[..., ::-1]  # BGR->RGB
    #  primary_image = primary_image[..., ::-1]  # BGR->RGB
    primary_image = cv2.resize(primary_image, (224, 224))
    # print(primary_image)
    pose = data.get('pose', None)
    gripper_val = data.get('gripper', None)
    pose= data.get('pose', None)

    
    # 计算直方图（例如划分为50个区间）
    # hist, bin_edges = np.histogram(dep_primary_image.flatten(), bins=50)
    # print("直方图统计:")
    # print("  Bin 边界:", bin_edges)
    # print("  每个区间内像素数量:", hist)


    gripper = data.get('gripper', None)
    if gripper == True:
        gs = [1.0]
    else:
        gs = [0.0]
    n = [0.0]
    proprio = pose + n + gs

    proprio = list(pose) + n + gs
    lang = "put the white cube into the box"
    # Grab image input & format prompt
    instruction =  f"What action should the robot take to {lang}, given robot proprioception {proprio}?"
    print(instruction)
    image = Image.fromarray(primary_image)
    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(instruction, image, return_tensors="pt").to("cuda:0",dtype=torch.bfloat16)
    # print("Processed inputs keys:", list(inputs.keys()))
    # for key, tensor in inputs.items():
    #     if hasattr(tensor, 'shape'):
    #         # 打印每个输入 tensor 的形状以及数值范围，方便检查深度信息是否处理正常
    #         try:
    #             print(f"Key: {key}, shape: {tensor.shape}, min: {tensor.min().item()}, max: {tensor.max().item()}")
    #         except Exception as e:
    #             print(f"Key: {key}, shape: {tensor.shape} (无法获取数值范围：{e})")
    
    # # 假设 processor 在处理深度图像时会在返回字典中保留 'depth_images' 键，
    # # 如果存在这一键，则可以对比数值范围或标准化效果
    # if "depth_pixel_values" in inputs:
    #     depth_tensor = inputs["depth_pixel_values"]
    #     # 正确转换方式：
    #     numpy_array = depth_tensor.to(dtype=torch.float32).cpu().numpy()
    #     print_depth_statistics(numpy_array)

    action = vla.predict_action(**inputs, unnorm_key="ur5", do_sample=False)
    # Execute...
    print(action)
    # with torch.no_grad():
    #     model_out = vla(**inputs)
    #     print(model_out)