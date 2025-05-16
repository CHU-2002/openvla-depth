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

- [ur5_robo_dataset2](https://huggingface.co/datasets/CHU-2002/ur5_robo_dataset)  
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

## 🔧 Fine-tuning Workflow

Follow these steps to fine-tune the model with your own data:

### 1. Prepare Dataset

You can either:

- Follow the instructions from [UR5 Dataset Builder](https://github.com/CHU-2002/UR5_dataset_builder) to build a suitable dataset, **or**
- Directly download the preprocessed dataset from Hugging Face: [ur5_robo_dataset](https://huggingface.co/datasets/CHU-2002/ur5_robo_dataset/)

### 2. Register Dataset

Ensure the dataset is registered inside the following files (this is already done for `ur5_robo_dataset`):

- `/prismatic/vla/dataset/rlds/oxe/config.py`
- `/prismatic/vla/dataset/rlds/oxe/transforms.py`

### 3. Launch Fine-tuning

Use `vla-scripts/finetune.py` for training. Example command for A100:

```bash
PYTHONPATH=$(pwd) torchrun --standalone --nnodes 1 --nproc-per-node 1 \
vla-scripts/finetune.py \
--data_root_dir  /path/to/dataset \
--run_root_dir /path/to/output_dir \
--adapter_tmp_dir /path/to/tmp_dir \
--lora_rank 32 \
--batch_size 8 \
--grad_accumulation_steps 1 \
--learning_rate 1e-4 \
--image_aug True \
--wandb_project your_project_name \
--wandb_entity your_wandb_team \
--save_steps 25000 \
--max_steps 25000
```
Example for RTX 4090:

```bash
PYTHONPATH=$(pwd) torchrun --standalone --nnodes 1 --nproc-per-node 1 \
vla-scripts/finetune.py \
--data_root_dir  /path/to/dataset \
--dataset_name ur5_robo_dataset \
--run_root_dir /path/to/output_dir \
--adapter_tmp_dir  /path/to/tmp_dir \
--lora_rank 32 \
--batch_size 1 \
--grad_accumulation_steps 16 \
--learning_rate 1e-4 \
--image_aug True \
--wandb_project your_project_name \
--wandb_entity your_wandb_team \
--save_steps 12000 \
--max_steps 12000
```
Adjust the parameters accordingly to your hardware setup and training plan.

### 4. Update Inference Configuration

Modify `openvla_exp.py` to set the model checkpoint path and dataset statistics (mean/std) for evaluation or robotic control.

### 5. Run Inference

Use the `test_reasoning.py` script to validate the fine-tuned model's reasoning and control performance.

## Resources

- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA Website](https://openvla.github.io)
- [OpenVLA-7B on Hugging Face](https://huggingface.co/openvla/openvla-7b)

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

For feedback, issues, or contributions, please open an issue or submit a pull request.

