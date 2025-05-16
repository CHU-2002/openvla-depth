# OpenVLA-Depth

**OpenVLA-Depth** extends the [OpenVLA](https://github.com/openvla/openvla) project by integrating depth information into the vision-language-action (VLA) pipeline. This extension enables more accurate spatial reasoning and manipulation in complex 3D environments, especially useful for robotic systems equipped with RGB-D sensors.

## Overview

OpenVLA is a family of generalist VLA models trained on large-scale robotic datasets. OpenVLA-Depth builds on this architecture by incorporating depth inputs into both the model and processor, enabling better performance on tasks that require 3D understanding.

## Usage

This repository follows the **same usage conventions as the original [OpenVLA](https://github.com/openvla/openvla)**. All training, inference, and fine-tuning scripts can be used with minimal changes.

### Pretrained and Fine-tuned Models

- [openvla-local](https://huggingface.co/CHU-2002/openvla-local)  
- [openvla-finetune](https://huggingface.co/CHU-2002/openvla-finetune)  
- [openvla-depth](https://huggingface.co/CHU-2002/openvla-depth)  

### Training Dataset

- [ur5_robo_dataset2](https://huggingface.co/datasets/CHU-2002/ur5_robo_dataset2)  
  This dataset contains multi-modal sequences suitable for UR5 robotic manipulation with RGB-D input.

### Inference and Utility Scripts

- `openvla_exp.py`: End-to-end inference pipeline for UR5 + RealSense setup.
- Additional scripts are provided for:
  - Inference testing
  - Data loading and preprocessing validation
  - Depth and RGB input pipeline verification

### Fine-tuning

Fine-tuning follows the **same procedure as OpenVLA**, with necessary extensions to handle depth features. Specifically:

- The model and processor are customized in:
  - `prismatic/extern/hf/modeling_prismatic.py`
  - `prismatic/extern/hf/processing_prismatic.py`

These modifications enable the model to accept and integrate depth images alongside RGB inputs during training and inference.

## Resources

- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA Website](https://openvla.github.io)
- [OpenVLA-7B on Hugging Face](https://huggingface.co/openvla/openvla-7b)

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

For feedback, issues, or contributions, please open an issue or submit a pull request.

