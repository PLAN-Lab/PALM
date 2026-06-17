#!/bin/bash
# LIBERO-10 Trajectory Data Collection Script
# Collects trajectory data from test phase using trained model
# Saves in libero_90_converted format

# Activate conda environment
source /data/yifan26/anaconda3/etc/profile.d/conda.sh
conda activate seer

# Set environment variables
export CUDA_VISIBLE_DEVICES=9
export MUJOCO_GL=egl

# Configuration parameters
CHECKPOINT_PATH="/data/yifan26/LiuYuanzhe/Seer/checkpoints/libero/38.pth"
OUTPUT_DIR="/data/yifan26/LiuYuanzhe/Seer/libero_10_rollouts_V3"
VIT_CHECKPOINT="/data/yifan26/LiuYuanzhe/Seer/checkpoints/mae_vit/mae_pretrain_vit_base.pth"
LOG_DIR="/data/yifan26/LiuYuanzhe/Seer/output/collection_logs"

# Create log directory
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${LOG_DIR}/collect_${TIMESTAMP}.log"

echo "================================================"
echo "LIBERO-10 Trajectory Data Collection"
echo "================================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Log File: ${LOGFILE}"
echo "================================================"

# Run collection script
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=10133 \
    /data/yifan26/LiuYuanzhe/Seer/collect_libero_data.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${VIT_CHECKPOINT} \
    --calvin_dataset "" \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 50000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 64 \
    --precision fp32 \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name collect_rollouts \
    --transformer_layers 24 \
    --phase "evaluate" \
    --finetune_type "libero_10" \
    --save_checkpoint_path "checkpoints/" \
    --action_pred_steps 3 \
    --future_steps 3 \
    --sequence_length 7 \
    --obs_pred \
    --gripper_width \
    --eval_libero_ensembling \
    --rollout_output_dir ${OUTPUT_DIR} \
    --resume_from_checkpoint ${CHECKPOINT_PATH} | tee ${LOGFILE}

echo ""
echo "================================================"
echo "Collection Complete! Data saved at: ${OUTPUT_DIR}"
echo "Log File: ${LOGFILE}"
echo "================================================"

