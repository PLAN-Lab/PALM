#!/bin/bash

# Activate palm conda environment
conda activate palm

export CUDA_VISIBLE_DEVICES=1
# Use EGL rendering (don't set PYOPENGL_PLATFORM!)
export MUJOCO_GL=egl

# Configuration for evaluating LIBERO_GOAL checkpoint
# Modify the checkpoint path and list below to match your trained models
pthlist=("sft_1" "sft_2" "sft_3" "sft_4" "sft_5" "sft_6" "sft_7" "sft_8" "sft_9" "sft_10")
for ckpt_id in "${pthlist[@]}"; do
    # Path to the directory containing finetuned checkpoints
    resume_from_checkpoint="xxx/libero_goal_finetune"
    vit_checkpoint_path="xxx/mae_pretrain_vit_base.pth"
    this_resume_from_checkpoint="${resume_from_checkpoint}/${ckpt_id}.pth"
    save_checkpoint_path="checkpoints/"
    dirname=$(basename "$resume_from_checkpoint")
    LOG_DIR="xxx/eval_logs/${dirname}"
    mkdir -p ${LOG_DIR}
    test_id="${ckpt_id}"
    logfile="${LOG_DIR}/${test_id}.log"

    node=1
    node_num=1

    python -m torch.distributed.run  --nnodes=${node} --nproc_per_node=${node_num} --master_port=10163 eval_libero.py \
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset "" \
        --libero_path "xxx/LIBERO" \
        --workers 16 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 20 \
        --seed 42 \
        --batch_size 64 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --run_name test \
        --transformer_layers 24 \
        --phase "evaluate" \
        --finetune_type "libero_goal" \
        --save_checkpoint_path ${save_checkpoint_path} \
        --action_pred_steps 3 \
        --future_steps 3 \
        --sequence_length 7 \
        --obs_pred \
        --gripper_width \
        --eval_libero_ensembling \
        --resume_from_checkpoint ${this_resume_from_checkpoint} | tee ${logfile}
done


