## 🛠️ Environment Setup

We recommend using Anaconda to manage the environment.

```bash
conda create -n dda python=3.9
conda activate dda

# Install PyTorch (Please match your CUDA version)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install dependencies
pip install numpy pytorch-metric-learning
```

## 📂 Data Preparation

1. Training Data (MSCOCO & DDA-Aligned)
You need to download the training set [modelscope](https://modelscope.cn/datasets/roych1997/Dual_Data_Alignment/files) first.

Structure your dataset directory as follows:

```bash
/path/to/datasets/
├── real/         # Real images (e.g., COCO Train2017)
│   ├── 000000000009.jpg
│   └── ...
├── fake/         # DDA-aligned synthetic images
│   ├── 000000000009.png
│   └── ...
└── MSCOCO_train2017.json  # Quality metadata (Optional)
```

2. Configure Paths
Modify the train.sh file to point to your dataset paths:

```bash
# In train.sh
REAL_PATH="/path/to/train2017/real_0"
FAKE_PATH="/path/to/train2017/fake_1"
QUALITY_JSON="./MSCOCO_train2017.json"
```

## 🚀 Training

To train the DINOv2-LoRA model on DDA-aligned data, simply run the provided shell script.

```bash
bash train.sh -g 0 -a 4 -n "DDA_Experiment"
```

Parameters

- -g: GPU ID (default: 0)
- -a: Gradient Accumulation Steps (default: 4)
- -n: Experiment Name Suffix

Hyperparameters
Key hyperparameters are configured in train.sh:

- Model: DINOv2 ViT-L/14
- LoRA Rank: 8
- Batch Size: 16 (Effective Batch Size = 16 * Accumulation Steps)
- Learning Rate: 1e-4
- Image Size: 336x336
