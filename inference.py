import torch
import torch.nn as nn
from models.dinov2_models_lora import DINOv2ModelWithLoRA
from torchvision import transforms
from PIL import Image

# Check if multiple GPUs are available
if torch.cuda.device_count() < 8:
    print(f"Warning: Only {torch.cuda.device_count()} GPUs available, but 8 requested")
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = 8
print(f"Using {num_gpus} GPUs for training")

# Initialize model
model = DINOv2ModelWithLoRA(name="dinov2_vitl14", lora_rank=8, lora_alpha=1, lora_targets=None)

# Load checkpoint first
ckpt = "pretrained/ckpt.pth"

device = 'cuda:0'
checkpoint = torch.load(ckpt, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)

# Set up data parallel training
if num_gpus > 1:
    # Specify which GPUs to use (0 through num_gpus-1)
    gpu_ids = list(range(num_gpus))
    model = nn.DataParallel(model, device_ids=gpu_ids)
    print(f"Model wrapped with DataParallel using GPUs: {gpu_ids}")

# Set model to training mode for training
model.eval()

# Define the image transformations
test_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# Load the image using PIL
image_path = "test_data/img03551.jpg"
image = Image.open(image_path).convert("RGB")
input_image = test_transform(image).unsqueeze(0).to(device)
with torch.no_grad(): 
    output = model(input_image).sigmoid().flatten()
print("Model output:", output)
