
# CRITICAL: Set MuJoCo rendering backend at the VERY BEGINNING
# BEFORE any imports (including standard library os import)
import os
os.environ['MUJOCO_GL'] = 'egl'

# Suppress NVML warning (driver/library version mismatch won't affect training)
import warnings
warnings.filterwarnings('ignore', message="Can't initialize NVML")

import random
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed_utils import init_distributed_device, world_info_from_env

from utils.eval_utils_libero import eval_one_epoch_libero_ddp

# try:
#     from utils.eval_utils_libero import eval_one_epoch_libero_ddp as eval_one_epoch_calvin_ddp
# except:
#     pass 
# from utils.eval_utils_libero import eval_one_epoch_libero_ddp as eval_one_epoch_calvin_ddp
from torch.distributed.elastic.multiprocessing.errors import record
# from utils.arguments_utils import get_args_and_cfg
from utils.arguments_utils import get_parser
from pdb import set_trace
from models.palm_model import PALMAgent

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    parser = get_parser(is_eval=True)
    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)

    # model
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
        )

    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")

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
    model = model.to(device_id)
    model._init_model_type()
  
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        
        # Handle sequence_length mismatch between pretrain (11) and finetune/eval (7)
        # This happens when loading a pretrained checkpoint for evaluation
        if "module.transformer_backbone_position_embedding" in checkpoint["model_state_dict"]:
            ckpt_pe_shape = checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"].shape
            model_pe_shape = ddp_model.module.transformer_backbone_position_embedding.shape
            if ckpt_pe_shape != model_pe_shape:
                if args.rank == 0:
                    print(f"Adjusting position embedding from {ckpt_pe_shape} to {model_pe_shape}")
                checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"] = \
                    checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"][:, :args.sequence_length, :, :]
        
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)

    ddp_model.eval()
    
    eval_log_dir = 'evaluate'

    # Support all LIBERO suites: libero_10, libero_spatial, libero_object, libero_goal
    # if args.finetune_type == "libero_10":
    if "libero" in args.finetune_type:
        eval_one_epoch_libero_ddp(
            args=args,
            model=ddp_model,
            image_processor=model.image_processor,
            tokenizer=clip,
        )
    else:
        raise NotImplementedError

if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    main()
