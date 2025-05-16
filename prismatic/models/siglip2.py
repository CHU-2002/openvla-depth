# my_custom_vit.py (你本地新建的文件)
import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer
import timm

@register_model
def vit_gopt_siglip2_384(pretrained=False, **kwargs):
    """
    这里定义 ViT-gopt-16-SigLIP2-384 的结构:
    - 例如继承 VisionTransformer 并指定 patch_size, embed_dim 等
    - 和 timm 源码中 vit_base_patch16_384 做法类似
    """
    model_kwargs = dict(
        # 你要的参数，如 patch_size=16, embed_dim=..., depth=..., etc
        patch_size=16,
        img_size=384,
        # ...
    )
    model_kwargs.update(kwargs)
    model = VisionTransformer(**model_kwargs)

    # 如果 pretrained=True，则从本地或远程下载权重
    if pretrained:
        # 在这一步 load_state_dict, 例如从 huggingface_hub 下载
        pass

    return model

featurizer = timm.create_model(
    'vit_gopt_siglip2_384',  # 这个名字就是上面 @register_model 的函数名
    pretrained=False,
    num_classes=0,
)
