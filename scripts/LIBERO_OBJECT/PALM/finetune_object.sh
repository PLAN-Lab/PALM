#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8

### NEED TO CHANGE ###
save_checkpoint_path="xxx"
root_dir="xxx/PALM"
vit_checkpoint_path="xxx/mae_pretrain_vit_base.pth"
finetune_from_pretrained_ckpt="xxx/palm.pth"
libero_path="libero_object_converted"  # Path to converted LIBERO_OBJECT training data
### NEED TO CHANGE ###

calvin_dataset_path="calvin/dataset/task_ABC_D"
node=1
node_num=9

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=29503 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 5 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type libero_finetune \
    --root_dir ${root_dir} \
    --wandb_project libero_object_finetune \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name libero_object_finetune \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --loss_image \
    --loss_action \
    --reset_action_token \
    --reset_obs_token \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 25 \
    --gripper_width \
    --warmup_epochs 5 \
    --libero_path ${libero_path} \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    --report_to_wandb



