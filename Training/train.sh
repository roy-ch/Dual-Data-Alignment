#!/usr/bin/env bash
# Training Script for DINOv2-LoRA with Data Augmentation
# Usage:
#   bash train.sh -g 0 -c "RGB" -a 8 -n "experiment_name"

set -euo pipefail

# ========= User Configuration =========
# TODO: Update these paths for your environment
REAL_PATH="/xiejingyi/datasets/StyleCOCO/train2017/real"
FAKE_PATH="/xiejingyi/datasets/StyleCOCO/train2017/fake"
QUALITY_JSON="./MSCOCO_train2017.json"
CHECKPOINTS_DIR="./checkpoints"

# Experiment Settings
# ARCH="DINOv2-LoRA:dinov2_vitl14"
LORA_RANK=8
LORA_ALPHA=1
OPTIM="adam"
NITER=1
BATCH_SIZE=16
ACCUM_STEPS=4
CROP_SIZE=336
LEARNING_RATE=1e-4

# Augmentation Settings
# DOWN_RESIZE_FACTOR=0.2
# UPPER_RESIZE_FACTOR=3.5
# P_JPEG_FAKE=0.5
P_PIXELMIX=0.2
R_PIXELMIX=0.8
# METH_PIXELMIX="uniform"
P_FREQMIX=0.2
R_FREQMIX=0.8
# METH_FREQMIX="uniform"

# Advanced Features
# USE_CONTRASTIVE=true

# ========= Command Line Arguments =========
GPU_ID=0
EXP_SUFFIX=""

while getopts ":g:c:a:n:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG" ;;
    # c) MIX_COLOR_SPACE="$OPTARG" ;;
    a) ACCUM_STEPS="$OPTARG" ;;
    n) EXP_SUFFIX="$OPTARG" ;;
    # \?) echo "Usage: $0 [-g GPU_ID] [-c MIX_COLOR_SPACE] [-a ACCUM_STEPS] [-n EXP_SUFFIX]"; exit 1 ;;
  esac
done

# ========= Setup Flags & Name =========
OPT_FLAGS=""
# if $USE_CONTRASTIVE; then OPT_FLAGS+=" --contrastive"; fi

# EXP_NAME="DINO_${CROP_SIZE}_LoRA${LORA_RANK}_LR${LEARNING_RATE}_BS${BATCH_SIZE}_${MIX_COLOR_SPACE}"
EXP_NAME="DINO_${CROP_SIZE}_LoRA${LORA_RANK}_LR${LEARNING_RATE}_BS${BATCH_SIZE}"
if [[ -n "${EXP_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

echo ">>> Starting Training: ${EXP_NAME}"
# echo ">>> GPU: ${GPU_ID} | Accumulation: ${ACCUM_STEPS} | Color: ${MIX_COLOR_SPACE}"
echo ">>> GPU: ${GPU_ID} | Accumulation: ${ACCUM_STEPS}"

# python train.py \
#   --gpu_ids "${GPU_ID}" \
#   --name "${EXP_NAME}" \
#   --cropSize "${CROP_SIZE}" \
#   --real_image_dir "${REAL_PATH}" \
#   --vae_image_dir "${FAKE_PATH}" \
#   # --arch "${ARCH}" \
#   --batch_size "${BATCH_SIZE}" \
#   --lr "${LEARNING_RATE}" \
#   --accumulation_steps "${ACCUM_STEPS}" \
#   --optim "${OPTIM}" \
#   --niter "${NITER}" \
#   --lora_rank "${LORA_RANK}" \
#   --lora_alpha "${LORA_ALPHA}" \
#   --vae_models "${VAE_PATH}" \
#   # --p_jpeg_fake "${P_JPEG_FAKE}" \
#   --checkpoints_dir "${CHECKPOINTS_DIR}" \
#   # --down_resize_factors "${DOWN_RESIZE_FACTOR}" \
#   # --upper_resize_factors "${UPPER_RESIZE_FACTOR}" \
#   --quality_json "${QUALITY_JSON}" \
#   --p_pixelmix "${P_PIXELMIX}" \
#   --r_pixelmix "${R_PIXELMIX}" \
#   # --meth_pixelmix "${METH_PIXELMIX}" \
#   --p_freqmix "${P_FREQMIX}" \
#   --r_freqmix "${R_FREQMIX}" \
#   # --meth_freqmix "${METH_FREQMIX}" \
#   $OPT_FLAGS
python train.py \
  --gpu_ids "${GPU_ID}" \
  --name "${EXP_NAME}" \
  --cropSize "${CROP_SIZE}" \
  --real_image_dir "${REAL_PATH}" \
  --vae_image_dir "${FAKE_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --accumulation_steps "${ACCUM_STEPS}" \
  --optim "${OPTIM}" \
  --niter "${NITER}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --quality_json "${QUALITY_JSON}" \
  --p_pixelmix "${P_PIXELMIX}" \
  --r_pixelmix "${R_PIXELMIX}" \
  --p_freqmix "${P_FREQMIX}" \
  --r_freqmix "${R_FREQMIX}" 

echo ">>> Training finished. Weights saved to: ${CHECKPOINTS_DIR}/${EXP_NAME}"