import glob
import os
import random

# Suppress NVML warning (driver/library version mismatch won't affect training)
import warnings
warnings.filterwarnings('ignore', message="Can't initialize NVML")
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from models.palm_model import PALMAgent
from utils.train_utils import get_checkpoint, train_one_epoch_calvin, get_ckpt_name
from utils.arguments_utils import get_parser
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset, get_droid_dataset, get_libero_pretrain_dataset, get_libero_finetune_dataset, get_real_finetune_dataset, get_real_val_dataset, get_oxe_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  

def get_mask_shape_from_dataset(args):
    """
    Get mask dimensions from dataset
    
    Args:
        args: Training arguments
        
    Returns:
        tuple: (mask_height, mask_width)
    """
    try:
        print("Inferring mask dimensions from dataset...")
        
        if args.finetune_type == "droid":
            # For DROID dataset, check filesystem directly
            import glob
            import numpy as np
            
            # Find first available mask file
            mask_pattern = f"{args.root_dir}/droid_language_v2/episodes/*/steps/*/mask.npz"
            mask_files = glob.glob(mask_pattern)
            
            if mask_files:
                mask_data = np.load(mask_files[0])
                mask_shape = mask_data['mask'].shape  # (H, W)
                print(f"Detected mask dimensions from {mask_files[0]}: {mask_shape}")
                return mask_shape
            else:
                print(f"Warning: No mask files found, using default dimensions (180, 320)")
                return (180, 320)
        else:
            print(f"Warning: Non-DROID dataset ({args.finetune_type}), using default mask dimensions (180, 320)")
            return (180, 320)
            
    except Exception as e:
        print(f"Warning: Failed to get mask dimensions ({e}), using default dimensions (180, 320)")
        return (180, 320)

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params

def validate_one_epoch(args, model, val_loader, device_id, epoch, wandb):
    """
    Validate the model for one epoch on the validation set
    """
    model.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_image_loss = 0.0
    total_gripper_loss = 0.0
    num_batches = 0
    
    print(f"\n{'='*80}")
    print(f"Validation Epoch {epoch+1}")
    print(f"{'='*80}")
    
    cast_dtype = torch.bfloat16
    
    with torch.no_grad():
        for batch_idx, batch_calvin in enumerate(val_loader):
            try:
                # Process data exactly the same way as in train_utils.py
                # images
                images_primary = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
                images_wrist = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
                # text tokens
                text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
                
                # states
                states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
                if args.gripper_width:
                    input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
                else:
                    input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
                    input_states[..., 6:] = (input_states[..., 6:] + 1) // 2
                
                # actions
                actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
                actions[..., 6:] = (actions[..., 6:] + 1) // 2
                
                input_image_primary = images_primary[:, :args.sequence_length, :]
                input_image_wrist = images_wrist[:, :args.sequence_length, :]
                input_text_token = text_tokens[:, :args.sequence_length, :]
                input_state = input_states[:, :args.sequence_length, :]
                
                # Create dummy auxiliary data (masks, contacts, spatials, trajectories)
                batch_size, seq_len = images_primary.shape[:2]
                input_mask_primary = torch.zeros(batch_size, args.sequence_length, 180, 320, device=device_id, dtype=cast_dtype)
                input_mask_wrist = torch.zeros(batch_size, args.sequence_length, 180, 320, device=device_id, dtype=cast_dtype)
                input_contact_primary = torch.full((batch_size, args.sequence_length, 2), 0.5, device=device_id, dtype=cast_dtype)
                input_contact_wrist = torch.full((batch_size, args.sequence_length, 2), 0.5, device=device_id, dtype=cast_dtype)
                input_spatial_primary = torch.full((batch_size, args.sequence_length, 2), 0.5, device=device_id, dtype=cast_dtype)
                input_spatial_wrist = torch.full((batch_size, args.sequence_length, 2), 0.5, device=device_id, dtype=cast_dtype)
                input_traj_static = torch.zeros(batch_size, args.sequence_length, 784, 2, device=device_id, dtype=cast_dtype)
                input_visibility_static = torch.zeros(batch_size, args.sequence_length, 784, device=device_id, dtype=cast_dtype)
                input_traj_gripper = torch.zeros(batch_size, args.sequence_length, 784, 2, device=device_id, dtype=cast_dtype)
                input_visibility_gripper = torch.zeros(batch_size, args.sequence_length, 784, device=device_id, dtype=cast_dtype)
                
                # Forward pass - consistent with train_utils.py
                (arm_pred_action, gripper_pred_action, progress_pred, loss_dit_diffusion, image_pred, arm_pred_state, gripper_pred_state, 
                 loss_arm_action, mask_pred_primary, mask_pred_wrist, contact_pred_primary, contact_pred_wrist, 
                 spatial_pred_primary, spatial_pred_wrist, traj_pred_primary, traj_pred_wrist) = model(
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                    action=actions[:, :args.sequence_length, :],
                    progress=None,  # Progress GT not available yet
                    mask_primary=input_mask_primary,
                    mask_wrist=input_mask_wrist,
                    contact_primary=input_contact_primary,
                    contact_wrist=input_contact_wrist,
                    spatial_primary=input_spatial_primary,
                    spatial_wrist=input_spatial_wrist,
                    traj_static=input_traj_static,
                    visibility_static=input_visibility_static,
                    traj_gripper=input_traj_gripper,
                    visibility_gripper=input_visibility_gripper,
                )
                
                # Compute loss - loss_arm_action is already included in return values
                action_loss = loss_arm_action.mean() if torch.is_tensor(loss_arm_action) else 0.0
                total_action_loss += action_loss.item() if torch.is_tensor(action_loss) else action_loss
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: action_loss={action_loss:.4f}")
                    
            except Exception as e:
                import traceback
                print(f"Error in validation batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
    
    # Calculate average loss
    if num_batches > 0:
        avg_action_loss = total_action_loss / num_batches
        
        print(f"\n{'='*80}")
        print(f"Validation Results (Epoch {epoch+1})")
        print(f"{'='*80}")
        print(f"  Action Loss:  {avg_action_loss:.4f}")
        print(f"  Valid Batches: {num_batches}")
        print(f"{'='*80}\n")
        
        # Log to wandb
        if args.rank == 0 and args.report_to_wandb and wandb is not None:
            wandb.log({
                "val/action_loss": avg_action_loss,
                "epoch": epoch,
            })
        
        return avg_action_loss
    else:
        print("Warning: No valid validation batches!")
        return float('inf')

@record
def main(args):
    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.save_checkpoint_path)}"
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)
    ptbs = args.world_size * args.batch_size * args.gradient_accumulation_steps
    print("training batch size:", ptbs)
    args.run_name = args.run_name.replace("PALM", f"PALM_ptbs{ptbs}_{args.transformer_layers}layers_{args.transformer_heads}heads_hd{args.hidden_dim}")
    print("run_name:", args.run_name)
    
    # Get mask dimensions if global affordance prediction is enabled
    mask_height, mask_width = get_mask_shape_from_dataset(args) if args.global_pred else (180, 320)
    
    model = PALMAgent(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
        global_pred=args.global_pred,
        global_pred_steps=args.global_pred_steps,
        mask_height=mask_height,
        mask_width=mask_width,
        local_pred=args.local_pred,
        local_pred_steps=args.local_pred_steps,
        spatial_pred=args.spatial_pred,
        spatial_pred_steps=args.spatial_pred_steps,
        spatial_num_candidates=getattr(args, "spatial_num_candidates", 10),
        dynamic_pred=args.dynamic_pred,
        dynamic_pred_steps=args.dynamic_pred_steps,
        dynamic_patch_size=args.dynamic_patch_size,
        no_pred_gripper_dynamic=args.no_pred_gripper_dynamic,
        no_unshuffle=args.no_unshuffle,
        flow_as_mask=args.flow_as_mask,
        # DiT decoder and progress prediction parameters
        use_dit_decoder=args.use_dit_decoder,
        predict_progress=args.predict_progress,
        dit_hidden_size=args.dit_hidden_size,
        dit_depth=args.dit_depth,
        dit_num_heads=args.dit_num_heads,
        dit_diffusion_steps=args.dit_diffusion_steps,
        dit_sampling_steps=args.dit_sampling_steps,
        dit_noise_schedule=args.dit_noise_schedule,
    )
    if args.finetune_type == "calvin":
        calvin_dataset = get_calvin_dataset(args, model.image_processor, clip, epoch=0, except_lang=args.except_lang)
    elif args.finetune_type == "droid":
        calvin_dataset = get_droid_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_pretrain":
        calvin_dataset = get_libero_pretrain_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_finetune":
        calvin_dataset = get_libero_finetune_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "real":
        calvin_dataset = get_real_finetune_dataset(args, model.image_processor, clip, epoch=0)
        # Load validation dataset for real-world data
        if hasattr(args, 'enable_validation') and args.enable_validation:
            try:
                calvin_val_dataset = get_real_val_dataset(args, model.image_processor, clip, epoch=0)
                print(f"✅ Validation dataset loaded: {calvin_val_dataset.dataloader.num_samples} samples")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load validation dataset: {e}")
                print(f"   Training will continue without validation.")
                calvin_val_dataset = None
        else:
            calvin_val_dataset = None
    elif args.finetune_type == "oxe":
        calvin_dataset = get_oxe_dataset(args, model.image_processor, clip, epoch=0)
    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        print("wandb_project :", args.wandb_project)
        print("wandb_entity :", args.wandb_entity)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    elif args.precision == "fp32":
        model = model.float()
        if 'vision_encoder' in args.bf16_module:
            model.vision_encoder.bfloat16()
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()
    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    total_params, trainable_params = count_parameters(model)
    print("total_params: {} M".format(total_params/1024/1024))
    print("trainable_params: {} M".format(trainable_params/1024/1024))
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)  # TODO make sure the parameters which need to be optimized are passing
    total_training_steps = calvin_dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = calvin_dataset.dataloader.num_batches * args.warmup_epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    if args.lr_scheduler == "linear":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == "cosine":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    resume_from_epoch = 0
    if args.finetune_from_pretrained_ckpt is not None:
        if args.rank == 0:
            print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_pretrained_ckpt}")
        checkpoint = torch.load(args.finetune_from_pretrained_ckpt, map_location="cpu")

        # Match keys by both legacy ("mask"/"contact"/"traj") and current
        # ("global"/"local"/"dynamic") naming conventions so that --reset_*
        # flags actually delete the corresponding weights regardless of which
        # convention the checkpoint was saved under.
        sd = checkpoint["model_state_dict"]
        image_decoder_keys = [k for k in sd if "image_decoder" in k]
        projector_keys = [k for k in sd if "projector" in k]
        action_decoder_keys = [k for k in sd if "action_decoder" in k]
        # global affordance (paper's "global" head; legacy name was "mask")
        global_decoder_keys = [k for k in sd if any(s in k for s in [
            "global_decoder", "global_primary_pred", "global_wrist_pred",
            "mask_decoder", "mask_primary_pred", "mask_wrist_pred",
        ])]
        # local affordance (paper's "local" head; legacy name was "contact")
        local_decoder_keys = [k for k in sd if any(s in k for s in [
            "local_decoder", "local_decoder_projector", "local_decoder_norm",
            "local_primary_pred", "local_wrist_pred",
            "contact_decoder", "contact_decoder_projector", "contact_decoder_norm",
            "contact_primary_pred", "contact_wrist_pred",
        ])]
        spatial_decoder_keys = [k for k in sd if any(s in k for s in [
            "spatial_decoder", "spatial_primary_pred", "spatial_wrist_pred",
        ])]
        # dynamic affordance (paper's "dynamic" head; legacy name was "traj")
        dynamic_decoder_keys = [k for k in sd if any(s in k for s in [
            "dynamic_decoder", "dynamic_decoder_obs_pred_projector",
            "dynamic_decoder_norm", "dynamic_decoder_pred",
            "dynamic_mask_token", "dynamic_decoder_position_embedding",
            "traj_decoder", "traj_mask_token",
            "traj_decoder_pred", "traj_decoder_obs_pred_projector",
        ])]

        def _maybe_del(keys):
            for k in list(keys):
                if k in sd:
                    del sd[k]

        if args.reset_action_token and "module.action_pred_token" in sd:
            del sd["module.action_pred_token"]
        if args.reset_obs_token and "module.obs_tokens" in sd:
            del sd["module.obs_tokens"]
        if args.reset_mask_token and "module.mask_token" in sd:
            del sd["module.mask_token"]
        # local/contact pred token name has changed; cover both
        if args.reset_contact_token:
            for nm in ["module.contact_pred_token", "module.local_pred_token"]:
                if nm in sd:
                    del sd[nm]
        if args.reset_spatial_token and "module.spatial_pred_token" in sd:
            del sd["module.spatial_pred_token"]
        # dynamic/traj pred token: cover both
        if args.reset_trajectory_token:
            for nm in ["module.traj_pred_token", "module.dynamic_pred_token"]:
                if nm in sd:
                    del sd[nm]

        if args.reset_image_decoder:
            _maybe_del(image_decoder_keys)
        if args.reset_action_decoder:
            _maybe_del(action_decoder_keys)
        if args.reset_mask_decoder:
            _maybe_del(global_decoder_keys)
        if args.reset_contact_decoder:
            _maybe_del(local_decoder_keys)
        if args.reset_spatial_decoder:
            _maybe_del(spatial_decoder_keys)
        if args.reset_trajectory_decoder:
            _maybe_del(dynamic_decoder_keys)

        # Sequence-length resize for the time positional embedding.
        pe_key = "module.transformer_backbone_position_embedding"
        if pe_key in sd and sd[pe_key].shape != ddp_model.module.transformer_backbone_position_embedding.shape:
            sd[pe_key] = sd[pe_key][:, :args.sequence_length, :, :]

        # Drop any key whose shape no longer matches the current model. This
        # is necessary because (a) we changed the spatial/local/dynamic head
        # output dims, and (b) older checkpoints may have been saved before
        # those refactors. Without this, load_state_dict throws a hard error
        # even with strict=False on same-name shape mismatches.
        cur_state = ddp_model.state_dict()
        dropped_for_shape = []
        for k, v in list(sd.items()):
            if k in cur_state and cur_state[k].shape != v.shape:
                dropped_for_shape.append((k, tuple(v.shape), tuple(cur_state[k].shape)))
                del sd[k]
        if args.rank == 0 and dropped_for_shape:
            print(f"[checkpoint] {len(dropped_for_shape)} keys dropped due to shape mismatch (will be reinitialised):")
            for k, ck_s, cur_s in dropped_for_shape[:20]:
                print(f"   - {k}: ckpt {ck_s} vs model {cur_s}")
            if len(dropped_for_shape) > 20:
                print(f"   ... ({len(dropped_for_shape) - 20} more)")

        if args.rank == 0:
            print(f"[checkpoint] loading {len(sd)} compatible parameters from pretrained ckpt")
        ddp_model.load_state_dict(sd, False)
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    ckpt_dir = os.path.join(f"{args.save_checkpoint_path}", args.run_name)
    if args.rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    ddp_model.train()
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader
        
        # Training
        train_one_epoch_calvin(
            args=args,
            model=ddp_model,
            epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calvin_loader=calvin_loader,
            device_id=device_id,
            wandb=wandb,
        )
        
        # Validation (if enabled for real-world data)
        if args.finetune_type == "real" and hasattr(args, 'enable_validation') and args.enable_validation:
            if 'calvin_val_dataset' in locals() and calvin_val_dataset is not None:
                calvin_val_dataset.set_epoch(epoch)
                val_loader = calvin_val_dataset.dataloader
                val_loss = validate_one_epoch(
                    args=args,
                    model=ddp_model,
                    val_loader=val_loader,
                    device_id=device_id,
                    epoch=epoch,
                    wandb=wandb,
                )
                # Resume to training mode
                ddp_model.train()
        
        # Save checkpoint at the end of each epoch as sft_{epoch+1}.pth
        if args.rank == 0 and args.save_checkpoint:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            # Save as sft_1.pth, sft_2.pth, sft_5.pth, sft_10.pth, etc.
            ckpt_name = f"sft_{epoch+1}.pth"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            print(f"Saving checkpoint to {ckpt_path}", flush=True)
            torch.save(checkpoint_dict, ckpt_path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    
