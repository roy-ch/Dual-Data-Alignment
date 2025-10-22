import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        """
        优化计算：直接 x @ A.T 再乘 B.T，避免计算 W_LoRA = B @ A
        """
        # 执行低秩变换
        lora_out = torch.einsum('...d, rd -> ...r', x, self.lora_A)  # x @ A.T
        lora_out = torch.einsum('...r, or -> ...o', lora_out, self.lora_B)  # (x @ A.T) @ B.T
        
        return lora_out * (self.alpha / self.rank)


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0, trainable_orig=False):
        super().__init__()
        self.original_layer = original_layer
        
        if not trainable_orig:
            for param in self.original_layer.parameters():
                param.requires_grad = False
                
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        
        if original_output.shape != lora_output.shape:
            raise ValueError(f"dimension mismatch: original output {original_output.shape}, LoRA output {lora_output.shape}")
            
        return original_output + lora_output
    
    def __getattr__(self, name):
        if name == 'weight':
            return self.original_layer.weight
        elif name == 'bias':
            return self.original_layer.bias
        else:
            return super().__getattr__(name)

def apply_lora_to_linear_layers(model, rank=4, alpha=1.0, target_modules=None, trainable_orig=False):
  
    for name, module in model.named_modules():
        if target_modules is not None and not any(target in name for target in target_modules):
            continue
            
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            parent = model if parent_name == '' else get_submodule(model, parent_name)
            
            setattr(parent, child_name, LoRALinear(module, rank, alpha, trainable_orig))
            
    return model

def get_submodule(model, submodule_name):
    """Helper function to get a submodule from a model"""
    if not submodule_name:
        return model
        
    parts = submodule_name.split('.')
    current_module = model
    
    for part in parts:
        if part.isdigit():  
            current_module = current_module[int(part)]
        else:  
            current_module = getattr(current_module, part)
            
    return current_module

def get_lora_params(model):
    lora_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_params.extend(module.parameters())
        elif isinstance(module, LoRALinear):
            lora_params.extend(module.lora.parameters())

    return lora_params