### NEED TO CHANGE ###
save_checkpoint_path="xxx/checkpoints"
root_dir="xxx/preprocess"
vit_checkpoint_path="xxx/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
### NEED TO CHANGE ###

### EXAMPLE ###
# - root_dir
#   - droid_success
#       - epsiodes
#           - 000000
#           - ......
#           - xxxxxx
#       - meta_info.h5
#       - shape_info.h5
### EXAMPLE ###

python slurm_train_intern.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 2 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset "" \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 20000 \
    --num_epochs 30 \
    --seed 42 \
    --batch_size 32 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type "droid" \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name mn_lang_droid \
    --save_checkpoint_path ${save_checkpoint_path} \
    --except_lang \
    --transformer_layers 24 \
    --phase "pretrain" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 11 \
    --window_size 11 \
    --future_steps 3 \
    --loss_action \
    --loss_image \
    --atten_goal 4 \
    --atten_goal_state \
    --atten_only_obs \
    --real_dataset_names "" \
    --root_dir ${root_dir} \
    --dataset_info  droid_success_languaged_0803 \
    --report_to_wandb \
    --warmup_epochs 3 \