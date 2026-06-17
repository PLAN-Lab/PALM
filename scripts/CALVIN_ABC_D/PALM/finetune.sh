#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
### NEED TO CHANGE ###
calvin_dataset_path="calvin/dataset/task_ABC_D"  # Using task_ABC_D dataset
save_checkpoint_path="checkpoints/"
finetune_from_pretrained_ckpt="calvin_pretrain/palm.pth"
vit_checkpoint_path="xxx/mae_pretrain_vit_base.pth"
### NEED TO CHANGE ###

node=1
node_num=9
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type "calvin" \
    --wandb_project "palm_calvin_finetune" \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name finetune_calvin_abc_d \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 10 \
    --future_steps 3 \
    --window_size 13 \
    --obs_pred \
    --loss_image \
    --loss_action \
    --report_to_wandb \
    --reset_action_token \
    --reset_obs_token \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt}
