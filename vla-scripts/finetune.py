"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch
import numpy as np
from PIL import Image
import time
import cv2
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_validation(vla, val_dataloader, device_id, action_tokenizer, current_step, val_time_limit):
    """
    在验证集上评估模型，计算平均损失与动作预测准确率，并通过 wandb 记录结果。

    参数：
      - vla: 待评估的 OpenVLAForActionPrediction 模型实例（包装了 Prismatic 模型的多模态前向逻辑）。
      - val_dataloader: 验证数据的 DataLoader，每个 batch 包含 'input_ids', 'attention_mask', 
                        'pixel_values', 'depth_pixel_values', 'labels' 等字段。
      - device_id: GPU 设备 ID。
      - action_tokenizer: 用于判断动作 token 的 tokenizer，其中 action_tokenizer.action_token_begin_idx 定义了
                          动作 token 起始值。
      - current_step: 当前训练步数，用于日志记录。
      - val_time_limit: 单次验证运行的时间上限（秒）。
    """
    vla.eval()
    val_losses = []
    val_action_accuracies = []
    start_time = time.time()

    # 为还原完整标签，需要知道视觉骨干插入了多少个 patch token
    with torch.no_grad():
        sample_batch = next(iter(val_dataloader))
        pixel_values_sample = sample_batch["pixel_values"].to(device_id).bfloat16()
        patch_features = vla.module.vision_backbone(pixel_values_sample)
        num_patches = patch_features.shape[1]

    with torch.no_grad():
        for batch in val_dataloader:
            if time.time() - start_time > val_time_limit:
                break

            # 将 batch 数据移动到设备并转换为 bfloat16
            input_ids = batch["input_ids"].to(device_id)
            attention_mask = batch["attention_mask"].to(device_id)
            # 显式使用 bfloat16() 确保数据类型正确，避免类型不匹配错误
            pixel_values = batch["pixel_values"].to(device_id).bfloat16()
            depth_values = batch["depth_pixel_values"].to(device_id).bfloat16()
            orig_labels = batch["labels"].to(device_id)  # shape: [B, L]
            
            # 前向计算：模型内部会构造完整的 multimodal 标签，
            # 格式为： [orig_labels[:, :1], 填充 IGNORE_INDEX 的 (B, num_patches), orig_labels[:, 1:]]
            output = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                depth_pixel_values=depth_values,
                labels=orig_labels,
            )
            loss = output.loss
            val_losses.append(loss.item())

            # 模型输出 logits 的 shape 应为 [B, L_full]，其中 L_full = L + num_patches
            preds = output.logits.argmax(dim=-1)  # shape: [B, L_full]
            
            # 重构完整标签：在原始标签中插入一段 IGNORE_INDEX (这里用 -100 表示)
            B, L = orig_labels.shape
            ignore_tensor = torch.full((B, num_patches), -100, dtype=orig_labels.dtype, device=orig_labels.device)
            full_labels = torch.cat([orig_labels[:, :1], ignore_tensor, orig_labels[:, 1:]], dim=1)

            # 仅统计有效位置：排除 IGNORE_INDEX，并假定只有大于 action_tokenizer.action_token_begin_idx 的 token 是动作 token
            valid_mask = (full_labels != -100) & (full_labels > action_tokenizer.action_token_begin_idx)

            if valid_mask.sum().item() > 0:
                correct = ((preds == full_labels) & valid_mask)
                acc = correct.sum().float() / valid_mask.sum().float()
                val_action_accuracies.append(acc.item())

    avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
    avg_acc = sum(val_action_accuracies) / len(val_action_accuracies) if val_action_accuracies else 0.0

    wandb.log({"val_loss": avg_loss, "val_action_accuracy": avg_acc}, step=current_step)
    print(f"[Step {current_step}] Validation: loss = {avg_loss:.4f}, action_accuracy = {avg_acc:.4f}")
    vla.train()
# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/home/yitongli/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 1                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 16                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 10000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = False                       # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                 # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on
    use_val_set: bool = False
    val_freq: int = 1000
    val_time_limit: int = 60 * 10  # 5 minutes


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    # import safetensors.torch

    # # 保存原始的 load_file 方法
    # original_load_file = safetensors.torch.load_file

    # def my_load_file(*args, **kwargs):
    #     state_dict = original_load_file(*args, **kwargs)
    #     print("\n[safetensors] Loaded state_dict keys:")
    #     for key in state_dict.keys():
    #         print(f"  - {key}")
    #     return state_dict

    # # 将 safetensors 的 load_file 方法替换为我们定义的 my_load_file
    # safetensors.torch.load_file = my_load_file


    
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
        ignore_mismatched_sizes=True, 
    )
    update_auto_map(cfg.vla_path)       # 更新 auto_map 指向本地实现
    check_model_logic_mismatch(cfg.vla_path) 
    dist.barrier()     
    # print(vla)
    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()


    # print("\n>>> Trainable parameter list:")
    # for name, param in vla.named_parameters():
    #     if param.requires_grad:
    #         print(" -", name)

    # # === 冻结部分参数 ===
    # # 冻结语言模型骨干，仅保持 lm_head 可训练
    # for name, param in vla.module.language_model.named_parameters():
    #     if "lm_head" not in name:
    #         param.requires_grad = False
    #     else:
    #         print(f"Keep training parameter: {name}")

    # # 冻结视觉骨干，确保其参数不更新
    # for name, param in vla.module.vision_backbone.named_parameters():
    #     param.requires_grad = False

    # # 保持 depth_backbone 和 projector 的参数为可训练状态（如果还未默认为 True）
    # for name, param in vla.module.depth_backbone.named_parameters():
    #     param.requires_grad = True
    # for name, param in vla.module.projector.named_parameters():
    #     param.requires_grad = True





    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    # print(vla)
    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        depth_transform=processor.image_processor.process_depth_image_hha,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
            train=False  # 设置为验证模式
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=PaddedCollatorForActionPrediction(
                processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
            ),
            num_workers=0
        )


    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)






    # if distributed_state.is_main_process:
    # # dataset_statistics 中可能包含：num_episodes, num_samples, action_dim, etc.
    #     stats = vla_dataset.dataset_statistics
    #     print("==== RLDS Dataset Statistics ====")
    #     for k, v in stats.items():
    #         print(f"{k}: {v}")
    #     # 保存到wandb
    #     wandb.log({f"dataset_stat/{k}": v for k, v in stats.items()})

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()

        for batch in dataloader:
            print("pixel:", batch["pixel_values"].shape)
            print("depth:", batch["depth_pixel_values"].shape)
            print("input_ids:", batch["input_ids"].shape)
            break

        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    depth_pixel_values=batch["depth_pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    output_projector_features=True,
                )
                loss = output.loss
                vis_tokens = output.projector_features
            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            N_vis = vis_tokens.shape[1] 
            action_logits = output.logits[:, N_vis : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            should_save_ckpt = (
                (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and  # 完整的梯度 step
                gradient_step_idx > 0 and
                gradient_step_idx % cfg.save_steps == 0
            )

            if should_save_ckpt:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    # base_vla = AutoModelForVision2Seq.from_pretrained(
                    #     cfg.vla_path, torch_dtype=torch.bfloat16, trust_remote_code=True
                    # )
                    # merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    # merged_vla = merged_vla.merge_and_unload()
                    merged_vla = vla.module.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()
                
            if cfg.use_val_set and gradient_step_idx > 0 and gradient_step_idx % cfg.val_freq == 0:
                if distributed_state.is_main_process:
                    print(f"Running validation at step {gradient_step_idx} ...")
                run_validation(vla, val_dataloader, device_id, action_tokenizer, gradient_step_idx, cfg.val_time_limit)

            # 停止训练条件



            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break
    
    # merged_vla = vla.module.merge_and_unload()
    # new_states = {
    # "action": {
    #     "mean": [
    #         0.00034084549406543374,
    #         -0.00030492557561956346,
    #         -0.0008138284320011735,
    #         0.016729077324271202,
    #         -1.8060404727293644e-07,
    #         2.659022015905066e-07,
    #         0.5941628217697144
    #     ],
    #     "std": [
    #         0.0036212557461112738,
    #         0.005531256087124348,
    #         0.0048943376168608665,
    #         4.131305694580078,
    #         0.00022504849766846746,
    #         0.0002433750923955813,
    #         0.49105530977249146
    #     ],
    #     "max": [
    #         0.01611638069152832,
    #         0.019588030874729156,
    #         0.018839538097381592,
    #         6.283182144165039,
    #         0.0011729226680472493,
    #         0.001676865853369236,
    #         1.0
    #     ],
    #     "min": [
    #         -0.019087255001068115,
    #         -0.01944923773407936,
    #         -0.01672576367855072,
    #         -6.2831830978393555,
    #         -0.0013649603351950645,
    #         -0.001678195665590465,
    #         0.0
    #     ],
    #     "q01": [
    #         -0.009785709381103515,
    #         -0.01367771863937378,
    #         -0.011039825081825256,
    #         -6.283165626525879,
    #         -0.0005680103786289692,
    #         -0.0005832317052409053,
    #         0.0
    #     ],
    #     "q99": [
    #         0.0100857949256897,
    #         0.01377965800464154,
    #         0.01329185485839849,
    #         6.283167839050293,
    #         0.0005469654710032047,
    #         0.0005947400396689773,
    #         1.0
    #     ],
    #     "mask": [
    #         True,
    #         True,
    #         True,
    #         True,
    #         True,
    #         True,
    #         False
    #     ]
    #     },
    #     "proprio": {
    #     "mean": [
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0
    #     ],
    #     "std": [
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0
    #     ],
    #     "max": [
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0
    #     ],
    #     "min": [
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0
    #     ],
    #     "q01": [
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0
    #     ],
    #     "q99": [
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0
    #     ]
    #     },
    #     "num_transitions": 9765,
    #     "num_trajectories": 97
    # }



    # merged_vla.norm_stats["ur5"] = new_states


    # num_test_samples = 150  # 或根据你的实际测试数据量设置
    # for i in range(num_test_samples):
    #     # 加载测试数据（必须是原来训练时存在的数据）
    #     data = np.load(f"task1/targ{i}.npy", allow_pickle=True).item()  # 注意这里使用了 .item() 来取出dict

    #     # 获取并处理图像 (BGR转RGB, resize)
    #     primary_image = data.get('primary_image', None)
    #     if primary_image is None:
    #         continue
    #     primary_image = primary_image[..., ::-1]  # BGR -> RGB
    #     primary_image = cv2.resize(primary_image, (224, 224))

    #     # 获取深度图像并resize（同样假设为单通道二维数组）
    #     dep_primary_image = data.get('primary_depth_image', None)
    #     if dep_primary_image is None:
    #         continue
    #     dep_primary_image = cv2.resize(dep_primary_image, (224, 224))

    #     # 获取机器人姿态及gripper值，构造 proprio 输入向量
    #     pose = data.get('pose', None)
    #     gripper_val = data.get('gripper', None)
    #     # 将gripper转换为0或1
    #     gs = [1.0] if gripper_val is True else [0.0]
    #     # 此处假设pose为列表或array；构造proprio：原始pose + placeholder (比如0.0) + gripper状态
    #     proprio = list(pose) + [0.0] + gs  # 这里 [0.0] 可用于占位，具体根据模型要求调整

    #     # 构造自然语言问题
    #     lang = "put the white cube into the box"
    #     instruction = f"What action should the robot take to {lang}, given robot proprioception {proprio}?"
    #     print(f"now is prediction on targ{i}.npy")
    #     print(instruction)

    #     # 利用 processor 将文本、RGB图像和深度图像打包成模型输入
    #     image = Image.fromarray(primary_image)
    #     depth_image = Image.fromarray(dep_primary_image)
    #     inputs = processor(instruction, image, depth_images=depth_image, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)
    #     # inputs = {k: v.to(device_id, dtype=torch.bfloat16) for k, v in inputs.items()}

    #     # 调用模型的 predict_action 方法，注意 unnorm_key（例如 "ur5"） 需与你之前注入的状态一致
    #     # 这里 do_sample=False 表示不随机抽样
    #     action = merged_vla.predict_action(**inputs, unnorm_key="ur5", do_sample=False)


    #     # 输出动作预测结果
    #     print("Predicted action:", action)

    

if __name__ == "__main__":
    finetune()