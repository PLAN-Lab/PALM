### NEED TO CHANGE ###
save_checkpoint_path="checkpoints/"
root_dir=""
real_dataset_names="" 
finetune_from_pretrained_ckpt="xxx/palm.pth"
vit_checkpoint_path="xxx/mae_pretrain_vit_base.pth" 

use_dit=false

# Build DiT-related extra args based on the switch.
if [ "${use_dit}" = "true" ]; then
    dit_args=(
        --use_dit_decoder
        --dit_hidden_size 384
        --dit_depth 12
        --dit_num_heads 12
        --dit_diffusion_steps 100
        --dit_sampling_steps 10
        --dit_noise_schedule squaredcos_cap_v2
        --predict_progress
        --loss_progress
        --loss_progress_ratio 0.1
        --reset_action_decoder
    )
else
    dit_args=()
fi

export CUDA_VISIBLE_DEVICES=8

node=1
node_num=1
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10214 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset "" \
    --workers 8 \
    --lr_scheduler cosine \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 8 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type real \
    --root_dir ${root_dir} \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name sn_ft \
    --save_checkpoint_path ${save_checkpoint_path} \
    --wandb_project "palm_droid_training" \
    --wandb_entity "yuanzhel-penn" \
    --transformer_layers 24 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --obs_pred \
    --loss_action \
    --loss_image \
    --warmup_epochs 5 \
    --real_dataset_names ${real_dataset_names} \
    --dataset_info droid_language_v2_train \
    --enable_validation \
    --real_val_dataset_names "droid_language_v2_val" \
    --reset_action_token \
    --reset_obs_token \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    --load_affordance \
    --reset_mask_decoder \
    --reset_contact_decoder \
    --reset_spatial_decoder \
    --reset_trajectory_decoder \
    --global_pred \
    --global_pred_steps 1 \
    --loss_global \
    --loss_global_ratio 0.1 \
    --local_pred \
    --local_pred_steps 1 \
    --loss_local \
    --loss_local_ratio 0.1 \
    --spatial_pred \
    --spatial_pred_steps 1 \
    --loss_spatial \
    --loss_spatial_ratio 0.1 \
    --dynamic_pred \
    --dynamic_pred_steps 1 \
    --loss_dynamic \
    --loss_dynamic_ratio 0.1 \
    --dynamic_patch_size 8 \
    --no_pred_gripper_dynamic \
    --no_unshuffle \
    "${dit_args[@]}"
    # --except_lang \
