"""
processing_prismatic.py

HuggingFace-style preprocessor definitions for Prismatic VLMs, inheriting from `ProcessorMixin`. Default configuration
specifies `siglip-224px+7b`.
"""

from typing import Any, ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
import math

# === Image Processing ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        """
        Initialize a PrismaticImageProcessor as a wrapper around a torchvision transform; this transform will be
        created by TIMM, and edited to follow our custom `image_resize_strategy` logic.
        @param use_fused_vision_backbone: Boolean indicating single or fused (dual) vision backbone
        @param image_resize_strategy: Prismatic image resize strategy in < resize-naive | resize-crop | letterbox >
        @param input_size: [TIMM :: `data_cfg`] Input image size as tuple (channels, width, height)
        @param interpolation: [TIMM :: `data_cfg`] Interpolation as string (default: "bicubic")
        @param mean: [TIMM :: `data_cfg`] Normalization mean as float tuple (or two-tuple if `fused_backbone`)
        @param std: [TIMM :: `data_cfg`] Normalization std as float tuple (or two-tuple if `fused_backbone`)
        """
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # Handle `None` default values
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        self.DEPTH_SCALE = 0.0001
        self.cx = 113.11272
        self.cy = 112.11167333333333
        self.fx = 134.58819499999998
        self.fy = 179.45092666666667
        self.h_max =  0.5
        self.k_disp = 10.0
        self.plane_normal= [
            0.03864950575371613,
            0.79421042555438,
            0.6064124138288431
        ]
        self.plane_offset = -0.36004510396370965

        # TIMM `data_cfg` Parameters
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # Grab torchvision transforms via TIMM =>> need to parse for specific "functional" transform values!
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
                crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`
                is_training=False,  # No image augmentations when loading the transform!
            )

            # [Validation] Ensure appropriate transform structure, expected sizes
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

            # HF Image Processors *must* be JSON-serializable; as such, cannot have torchvision. as an attribute.
            #   => Instead, we're going to parse the transform and call "torchvision.transforms.functional" (`tvf`)
            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            # Handle Prismatic `image_resize_strategy`
            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        # Dispatch **kwargs to super()
        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])"""
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0
        img_t = torch.vstack(imgs_t)

        return img_t
    
    def process_depth_image(self, image: Image.Image) -> torch.Tensor:
        """
        使用 min-max 归一化对单通道深度图像进行预处理，然后将其复制为三通道图像，
        以适应 ViT 输入要求。具体步骤包括：
        1. 如果需要 letterbox 填充，则先进行填充。
        2. 将原始 uint16 深度图转换为 PIL 的 32 位整型图像（"I"模式）。
        3. 采用与 RGB 图像相同的 Resize 和 CenterCrop 参数进行尺寸调整。
        4. 将调整后的图像转换为 tensor（形状为 [1, H, W]，float32）。
        5. 使用 min-max 方法对每张图像进行归一化：
            - 计算该图的最小值和最大值，
            - 将图像缩放到 [0, 1]，再映射至 [-1, 1]（可选）。
        6. 将归一化后的单通道 tensor 复制为三通道，形成 [3, H, W] 的输出。
        """
        # 第一步：如果需要 letterbox 填充，则先进行填充
        if self.tvf_do_letterbox:
            image = letterbox_pad_transform(image, self.tvf_letterbox_fill)
        
        # 第二步：将 uint16 深度图转换为 PIL 的 32 位整型图像（"I"模式）
        img = image.convert("I")
        
        # 第三步：使用与 RGB 相同的 Resize 和 CenterCrop 参数（此处仅使用第 0 套预处理参数）
        idx = 0
        img = TVF.resize(img, **self.tvf_resize_params[idx])
        img = TVF.center_crop(img, **self.tvf_crop_params[idx])
        
        # 第四步：转换为 tensor，结果形状为 [1, H, W]，且 TVF.to_tensor 会将像素值转换为 float32
        img_tensor = TVF.to_tensor(img)
        
        # 第五步：使用 min-max 方法对深度图像进行归一化
        # 先计算当前 tensor 的最小值和最大值
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        # 避免除零错误（若所有值相同时）
        if max_val - min_val > 0:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        else:
            img_tensor = img_tensor - min_val  # 全部置 0

        # 可选：将 [0,1] 的范围映射到 [-1,1]（这样能使均值更接近 0，有助于部分网络训练）
        img_tensor = img_tensor * 2 - 1
        
        # 第六步：复制单通道为三通道，以适应 ViT 的 3 通道输入要求
        img_tensor = img_tensor.repeat(3, 1, 1)
        
        return img_tensor

    def process_depth_image_hha(self, image: Image.Image) -> torch.Tensor:
        """
        将单通道深度图转换为 HHA 三通道张量:
        1. letterbox 填充（可选）
        2. PIL “I” 模式 → Resize → CenterCrop
        3. ToTensor 得到 [1,H,W] 原始深度
        4. 计算并归一化 Disparity, Height, Angle
        返回 [3, H, W]，通道顺序 [D, H, A]，范围 [0,1]
        """
        # 1) letterbox 填充
        if self.tvf_do_letterbox:
            image = letterbox_pad_transform(image, self.tvf_letterbox_fill)

        # 2) 转 32-bit、Resize+Crop
        img = image.convert("I")
        idx = 0
        img = TVF.resize(img, **self.tvf_resize_params[idx])
        img = TVF.center_crop(img, **self.tvf_crop_params[idx])

        # 3) ToTensor → [1,H,W]，值为原始 uint16（float32 表示）
        img_tensor = TVF.to_tensor(img)  # float32

        # 4.1) 深度米化
        depth_m = img_tensor * self.DEPTH_SCALE  # [1,H,W]

        # 4.2) Disparity 通道 (1/depth)
        disp = 1.0 / (depth_m + 1e-6)
        disp = torch.clamp(disp, 0.0, self.k_disp) / self.k_disp  # [1,H,W]

        # 准备坐标网格
        _, H, W = depth_m.shape
        device = depth_m.device
        ys = torch.arange(H, device=device).view(H, 1).expand(H, W)
        xs = torch.arange(W, device=device).view(1, W).expand(H, W)

        # 4.3) 反投影到相机坐标系
        Z = depth_m.squeeze(0)
        X = (xs - self.cx) / self.fx * Z
        Y = (ys - self.cy) / self.fy * Z

        # 4.4) Height 通道：点到平面的垂直距离
        # plane_normal·[X,Y,Z] + offset = signed distance
        pn = torch.tensor(self.plane_normal, device=device, dtype=torch.float32)
        d0 = float(self.plane_offset)
        dist = pn[0] * X + pn[1] * Y + pn[2] * Z + d0
        height = torch.abs(dist)
        height = torch.clamp(height, 0.0, self.h_max) / self.h_max  # [H,W]

        # 4.5) Angle 通道：表面法向量与平面法向量夹角
        # Sobel 差分 approximates depth gradients
        # 在 X 方向上做 sobel 差分
        dzdx = Z[:, 2:] - Z[:, :-2]           # [H, W-2]
        # 左右各复制一列
        dzdx = torch.cat([dzdx[:, :1], dzdx, dzdx[:, -1:]], dim=1)  # [H, W]

        # 在 Y 方向上做 sobel 差分
        dzdy = Z[2:, :] - Z[:-2, :]           # [H-2, W]
        # 上下各复制一行
        dzdy = torch.cat([dzdy[:1, :], dzdy, dzdy[-1:, :]], dim=0)  # [H, W]


        n_x = -dzdx
        n_y = -dzdy
        n_z = torch.ones_like(n_x)
        norm_n = torch.sqrt(n_x**2 + n_y**2 + n_z**2)
        n_x /= norm_n; n_y /= norm_n; n_z /= norm_n

        dot = n_x * pn[0] + n_y * pn[1] + n_z * pn[2]
        dot = torch.clamp(dot, -1.0, 1.0)
        angle = torch.acos(dot) / math.pi  # [H,W]

        # 5) 合并三通道
        hha = torch.stack([
            disp.squeeze(0),
            height,
            angle
        ], dim=0)  # [3, H, W]

        hha_vit = (hha - 0.5) / 0.5  

        return hha_vit

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we
        explicitly only handle PIL.Image.Image instances for simplicity.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray
        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # Return BatchFeature =>> note that for compatibility, constructor expects Dict[str, np.ndarray], so we convert
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        depth_images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        if not isinstance(depth_images, list):
            depth_images = [depth_images]
        depth_pixel_values = torch.stack([self.image_processor.process_depth_image(img) for img in depth_images])
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")
        
        if depth_pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Mismatch between number of text samples and depth images.")


        return BatchFeature(data={
            **text_inputs,
            "pixel_values": pixel_values,
            "depth_pixel_values": depth_pixel_values,
        })

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
