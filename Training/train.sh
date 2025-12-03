#!/usr/bin/env bash
# Training Script for DINOv2-LoRA with Data Augmentation
# Usage:
#   bash train.sh -g 0 -c "RGB" -a 8 -n "experiment_name"

set -euo pipefail

# ========= User Configuration =========
# TODO: Update these paths for your environment
REAL_PATH="/path/to/real/images"
FAKE_PATH="/path/to/vae/images"
QUALITY_JSON="./MSCOCO_train2017.json"
CHECKPOINTS_DIR="./checkpoints"

# Experiment Settings
LORA_RANK=8
LORA_ALPHA=1
OPTIM="adam"
NITER=1
BATCH_SIZE=16
ACCUM_STEPS=4
CROP_SIZE=336
LEARNING_RATE=1e-4

# Augmentation Settings
P_PIXELMIX=0.2
R_PIXELMIX=0.8
P_FREQMIX=0.2
R_FREQMIX=0.8

# ========= Command Line Arguments =========
GPU_ID=0
EXP_SUFFIX=""

while getopts ":g:c:a:n:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG" ;;
    a) ACCUM_STEPS="$OPTARG" ;;
    n) EXP_SUFFIX="$OPTARG" ;;
  esac
done

# ========= Setup Flags & Name =========
EXP_NAME="DINO_${CROP_SIZE}_LoRA${LORA_RANK}_LR${LEARNING_RATE}_BS${BATCH_SIZE}"
if [[ -n "${EXP_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

echo ">>> Starting Training: ${EXP_NAME}"
echo ">>> GPU: ${GPU_ID} | Accumulation: ${ACCUM_STEPS}"

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
