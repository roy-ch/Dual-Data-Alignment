from .dinov2_models import DINOv2Model
from .dinov2_models_lora import DINOv2ModelWithLoRA

VALID_NAMES = [
    "DINOv2:dinov2_vits14",
    "DINOv2:dinov2_vitb14",
    "DINOv2:dinov2_vitl14",
    "DINOv2:dinov2_vitg14",
    "DINOv2:dinov2_vitl14_register",
    "DINOv2-LoRA:dinov2_vits14",
    "DINOv2-LoRA:dinov2_vitb14",
    "DINOv2-LoRA:dinov2_vitl14",
    "DINOv2-LoRA:dinov2_vitg14",
    "DINOv2-LoRA:dinov2_vitl14_register",
]


def get_model(
    name, lora_rank=8, lora_alpha=1.0, lora_targets=None, huggingface_path=None
):

    assert (
        name in VALID_NAMES
    ), f"Invalid model name: {name}. Valid names: {VALID_NAMES}"

    print("\n" + "=" * 50)
    print(f"Initializing model: {name}")
    print(f"Parameters:")
    if "LoRA" in name:
        print(f"  - LoRA rank: {lora_rank}")
        print(f"  - LoRA alpha: {lora_alpha}")
        if lora_targets:
            print(f"  - LoRA targets: {lora_targets}")
        else:
            print(f"  - LoRA targets: default")
    elif "FullFinetune" in name:
        print(f"  - Training mode: Full Fine-tuning (all parameters)")

    if huggingface_path:
        print(f"  - HuggingFace path: {huggingface_path}")

    if name.startswith("DINOv2:"):
        model_family = "DINOv2"
        model_type = name[7:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        model = DINOv2Model(
            name=model_type,
        )

        print(f"  - Successfully initialized DINOv2 model")
        return model

    elif name.startswith("DINOv2-LoRA:"):
        model_family = "DINOv2 with LoRA"
        model_type = name[12:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        model = DINOv2ModelWithLoRA(
            name=model_type,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
        )

        print(f"  - Successfully initialized DINOv2-LoRA model")

        trainable_params = model.get_trainable_params()
        total_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"  - Total trainable parameters: {total_trainable_params:,}")

        return model

    else:
        raise ValueError(f"Unsupported model prefix in name: {name}")

    print("=" * 50 + "\n")
