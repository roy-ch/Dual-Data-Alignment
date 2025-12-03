import torch.nn as nn
from .dinov2_models import DINOv2Model


class DINOv2ModelWithLoRA(nn.Module):
    def __init__(
        self, name, num_classes=1, lora_rank=8, lora_alpha=1.0, lora_targets=None
    ):
        super(DINOv2ModelWithLoRA, self).__init__()

        self.base_model = DINOv2Model(
            name=name,
            num_classes=num_classes,
        )

        self.name = name
        try:
            from .lora import apply_lora_to_linear_layers, get_lora_params
        except ImportError:
            raise ImportError("LoRA module not found. Please check your installation.")

        if lora_targets is None:
            lora_targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

        print(f"Adding LoRA to DINOv2 (rank={lora_rank}, alpha={lora_alpha})")
        print(f"LoRA target modules: {lora_targets}")
        self.base_model.model = apply_lora_to_linear_layers(
            self.base_model.model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_targets,
            trainable_orig=False,
        )

        self._get_lora_params = lambda: get_lora_params(self.base_model.model)

    def get_trainable_params(self):

        lora_params = self._get_lora_params()
        fc_params = self.base_model.fc.parameters()
        total_lora_params = sum(p.numel() for p in lora_params)
        total_fc_params = sum(p.numel() for p in fc_params)
        return list(lora_params) + list(fc_params)

    def forward(self, x, return_feature=False, return_tokens=False):

        return self.base_model(
            x, return_feature=return_feature, return_tokens=return_tokens
        )

    def get_preprocessing_transforms(self):

        return self.base_model.get_preprocessing_transforms()
