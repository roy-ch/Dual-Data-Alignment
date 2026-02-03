import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models.dinov2_models_lora import DINOv2ModelWithLoRA
import torchvision.transforms.functional as F
import torch.nn.functional

from pytorch_metric_learning import losses


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt

        self.accumulation_steps = (
            opt.accumulation_steps if hasattr(opt, "accumulation_steps") else 1
        )
        self.current_step = 0

        lora_args = {}
        if hasattr(opt, "lora_rank"):
            lora_args["lora_rank"] = opt.lora_rank
        if hasattr(opt, "lora_alpha"):
            lora_args["lora_alpha"] = opt.lora_alpha

        lora_args["lora_targets"] = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

        self.model = DINOv2ModelWithLoRA(
            name="dinov2_vitl14",
            lora_rank=lora_args["lora_rank"],
            lora_alpha=lora_args["lora_alpha"],
            lora_targets=lora_args["lora_targets"],
        )

        torch.nn.init.normal_(self.model.base_model.fc.weight.data, 0.0, 0.02)

        if hasattr(self.model, "get_trainable_params"):
            params = self.model.get_trainable_params()
            print(
                "Training with LoRA - only LoRA and final layer parameters will be updated"
            )
        else:
            raise ValueError("LoRA model should have get_trainable_params method")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"\nTrainable parameters summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Trainable ratio: {trainable_params/total_params*100:.2f}%")

        print(f"\nTrainable parameter names:")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"  - {name}: {p.numel():,} parameters")

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, eta_min=opt.lr * 0.001, T_max=1000
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.contrastive_loss_fn = losses.ContrastiveLoss(
            pos_margin=0.0, neg_margin=1.0
        )

        if hasattr(opt, "device"):
            self.device = opt.device
        self.model.to(self.device)

    def set_input(self, input):
        components = ["real", "real_resized", "fake", "fake_resized"]

        input_stack = []
        for key in components:
            if input[key] is not None:
                input_stack.append(input[key])
        self.input = torch.cat(input_stack, dim=0).to(self.device)

        LABELS = {
            "real": 0,
            "real_resized": 0,
            "fake": 1,
            "fake_resized": 1,
        }
        label_stack = []
        for key in components:
            if input[key] is not None:
                label_stack += [LABELS[key]] * len(input[key])
        self.label = torch.tensor(label_stack).to(self.device).float()

    def forward(self):

        self.feature, self.output = self.model(self.input, return_feature=True)

        if hasattr(self.output, "view"):
            self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.current_step += 1
        self.forward()

        cls_loss = self.loss_fn(self.output.squeeze(1), self.label)

        contrastive_loss = self.contrastive_loss_fn(self.feature, self.label)
        
        total_loss = 0.5 * cls_loss + 0.5 * contrastive_loss

        self.loss = total_loss

        self.loss = self.loss / self.accumulation_steps
        
        self.loss.backward()

        if self.current_step % self.accumulation_steps == 0:

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def finalize_epoch(self):
        if self.current_step % self.accumulation_steps != 0:

            self.optimizer.step()
            self.optimizer.zero_grad()
