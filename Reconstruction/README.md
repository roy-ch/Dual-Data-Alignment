# VAE Batch Rebuilder

Batch encode and decode images using `diffusers` VAE models for constructing DDA-Training-Set.

## Quick Start

1. **Install Dependencies**

```code
pip install torch numpy diffusers pillow tqdm
```

2. **Prepare your COCO list file**

```code
find /path/to/your/MSCOCO/train2017 > coco.list
```

3. **Configuration** Edit following parameter in `vae_rec.py`

- **LIST_FILE**: Path to your COCO list file
- **OUTPUT_DIR**: Destination folder for results
- **VAE_MODEL_PATH**: Huggingface ID or your local path for VAE

4. **Run**

```code
python vae_rec.py
```
