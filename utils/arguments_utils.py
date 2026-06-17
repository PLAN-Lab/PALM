import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def get_parser(is_eval=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="RobotFlamingo",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # paper Table 8: 1e-4 pretrain / 1e-3 finetune
    parser.add_argument(
        "--lr_scheduler",
        default="cosine",
        type=str,
        help="constant, linear, or cosine. Paper Table 8 uses cosine decay.",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        default='/mnt/petrelfs/share_data/robomani/calvin_data/task_ABCD_D',
        help="path to calvin_dataset",
    )
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)  # paper Table 8: weight_decay = 1e-4
    # hot fix for torch.distributed.launch
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32", "bf16_and_fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=16)
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument('--rgb_pad', type=int, default=-1)
    parser.add_argument('--gripper_pad', type=int, default=-1)
    parser.add_argument(
        "--traj_cons",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--text_aug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--dif_ws",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--partial_data",
        default=False,
        action="store_true"
    )
    # data
    parser.add_argument("--save_every_iter", type=int, default=-1)
    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")
    # ceph
    parser.add_argument("--data_in_ceph",default=False, action="store_true")
    # oxe
    parser.add_argument("--root_dir", type=str, default="s3://real_data")
    parser.add_argument("--image_primary_size", type=int, default=200)
    parser.add_argument("--image_wrist_size", type=int, default=84)
    parser.add_argument("--finetune_type", type=str, default="",)   
    # save checkpoint
    parser.add_argument("--start_save_checkpoint", default=-1, type=int)
    parser.add_argument("--save_checkpoint", default=False, action="store_true")
    parser.add_argument("--save_checkpoint_path", required=True, type=str)
    parser.add_argument("--save_checkpoint_seq", type=int, default=1)
    # if validate
    parser.add_argument("--validation", default=False, action="store_true")
    # bf16 module
    parser.add_argument("--bf16_module", type=str, default="")
    # model structure 
    parser.add_argument("--sequence_length", type=int, default=7)  # paper Table 8: Historical Sequence Length = 7
    # for image prediction
    parser.add_argument("--future_steps", type=int, default=3)  # paper Table 8: Action Prediction Length = 3
    parser.add_argument("--num_resampler_query", type=int, default=9)
    parser.add_argument("--num_obs_token_per_image", type=int, default=9)
    parser.add_argument("--calvin_input_image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    # droid
    parser.add_argument("--primary_mode", type=str, default="image_primary")
    parser.add_argument("--small_size", type=int, default=0)
    parser.add_argument("--dataset_info", type=str, default="droid_success")
    # pretrain
    parser.add_argument("--finetune_from_pretrained_ckpt", type=str, default=None)
    # loss
    parser.add_argument("--loss_arm_action_ratio", type=float, default=1.0)
    parser.add_argument("--loss_gripper_action_ratio", type=float, default=0.01)
    parser.add_argument("--loss_global", action="store_true", help="Enable global affordance (mask) loss")
    parser.add_argument("--loss_global_ratio", type=float, default=0.1, help="Weight for global affordance loss")
    parser.add_argument("--loss_local", action="store_true", help="Enable local affordance (contact heatmap) loss")
    parser.add_argument("--loss_local_ratio", type=float, default=0.1, help="Weight for local affordance loss")
    parser.add_argument("--loss_spatial", action="store_true", help="Enable spatial affordance (placement points) loss")
    parser.add_argument("--loss_spatial_ratio", type=float, default=0.1, help="Weight for spatial affordance loss")
    parser.add_argument("--loss_dynamic", action="store_true", help="Enable dynamic affordance (trajectory) loss")
    parser.add_argument("--loss_dynamic_ratio", type=float, default=0.1, help="Weight for dynamic affordance loss")
    # Global Affordance prediction
    parser.add_argument("--global_pred", action="store_true", help="Enable global affordance prediction")
    parser.add_argument("--global_pred_steps", type=int, default=1, help="Number of future global affordance prediction steps")   
    parser.add_argument("--mask_height", type=int, default=180, help="Height of global affordance mask")
    parser.add_argument("--mask_width", type=int, default=320, help="Width of global affordance mask")
    # Local Affordance prediction
    parser.add_argument("--local_pred", action="store_true", help="Enable local affordance prediction")
    parser.add_argument("--local_pred_steps", type=int, default=1, help="Number of future local affordance prediction steps")
    # Spatial Affordance prediction
    parser.add_argument("--spatial_pred", action="store_true", help="Enable spatial affordance prediction")
    parser.add_argument("--spatial_pred_steps", type=int, default=1, help="Number of future spatial affordance prediction steps")
    parser.add_argument(
        "--spatial_num_candidates", type=int, default=10,
        help="Paper §3.3 Eq.6: spatial affordance predicts a SET of M candidate placement points per camera/step (M)."
    )
    # Dynamic Affordance prediction
    parser.add_argument("--dynamic_pred", action="store_true", help="Enable dynamic affordance prediction")
    parser.add_argument("--dynamic_pred_steps", type=int, default=1, help="Number of future dynamic affordance prediction steps")
    parser.add_argument("--dynamic_patch_size", type=int, default=8, help="Dynamic affordance patch size")
    parser.add_argument("--no_pred_gripper_dynamic", action="store_true", help="Disable wrist/gripper dynamic affordance prediction")
    parser.add_argument("--no_unshuffle", action="store_true", help="Do not use pixel_unshuffle for dynamic affordance")
    parser.add_argument("--flow_as_mask", action="store_true", help="Use optical flow as mask for image loss")
    # action_pred_steps
    parser.add_argument("--action_pred_steps", type=int, default=3)  # paper Table 7: Action Prediction Steps = 3
    
    # ========== DiT Decoder and Progress Prediction Parameters ==========
    # Action decoder type
    parser.add_argument(
        "--use_dit_decoder", 
        action="store_true", 
        default=False,
        help="Use DiT (Diffusion Transformer) for action decoding instead of MLP"
    )
    
    # Progress prediction (currently disabled, predict_progress will be False)
    parser.add_argument(
        "--predict_progress", 
        action="store_true", 
        default=False,
        help="Enable progress prediction (within-subtask completion estimation). Currently not used."
    )
    parser.add_argument(
        "--loss_progress", 
        action="store_true", 
        default=False,
        help="Enable progress prediction loss. Only effective when predict_progress is True."
    )
    parser.add_argument(
        "--loss_progress_ratio", 
        type=float, 
        default=0.1,
        help="Weight for progress prediction loss"
    )
    
    # DiT architecture parameters (Paper Table 7: hidden=384, depth=12, heads=12)
    parser.add_argument(
        "--dit_hidden_size",
        type=int,
        default=384,
        help="Hidden size for DiT decoder. Paper PALM Table 7 = 384."
    )
    parser.add_argument(
        "--dit_depth",
        type=int,
        default=12,
        help="Number of transformer layers in DiT. Paper PALM Table 7 = 12."
    )
    parser.add_argument(
        "--dit_num_heads",
        type=int,
        default=12,
        help="Number of attention heads in DiT. Paper PALM Table 7 = 12."
    )
    parser.add_argument(
        "--dit_diffusion_steps", 
        type=int, 
        default=100,
        help="Number of diffusion steps during training (for DDPM forward process)"
    )
    parser.add_argument(
        "--dit_sampling_steps", 
        type=int, 
        default=10,
        help="Number of sampling steps during inference (DDIM acceleration). Paper uses 10 steps."
    )
    parser.add_argument(
        "--dit_noise_schedule", 
        type=str, 
        default="squaredcos_cap_v2",
        choices=["linear", "squaredcos_cap_v2"],
        help="Noise schedule for diffusion process. 'squaredcos_cap_v2' is a cosine variant (recommended)."
    )
    # ====================================================================
    
    # obs_pred
    parser.add_argument("--obs_pred", default=False, action="store_true")
    parser.add_argument("--atten_only_obs", default=False, action="store_true")
    parser.add_argument("--attn_robot_proprio_state", default=False, action="store_true")
    parser.add_argument("--atten_goal", default=0, type=int)
    parser.add_argument("--atten_goal_state", default=False, action="store_true")
    # action mask ratio
    parser.add_argument("--mask_l_obs_ratio", default=0.00, type=float)
    # reset during finetuning
    parser.add_argument("--reset_action_token", default=False, action="store_true")
    parser.add_argument("--reset_obs_token", default=False, action="store_true")
    parser.add_argument("--reset_mask_token", default=False, action="store_true")
    parser.add_argument("--reset_image_decoder", default=False, action="store_true")
    parser.add_argument("--reset_action_decoder", default=False, action="store_true")
    parser.add_argument("--reset_mask_decoder", default=False, action="store_true", help="Reset mask decoder during finetuning")
    parser.add_argument("--reset_contact_token", default=False, action="store_true", help="Reset contact tokens during finetuning")
    parser.add_argument("--reset_contact_decoder", default=False, action="store_true", help="Reset contact decoder during finetuning")
    parser.add_argument("--reset_spatial_token", default=False, action="store_true", help="Reset spatial tokens during finetuning")
    parser.add_argument("--reset_spatial_decoder", default=False, action="store_true", help="Reset spatial decoder during finetuning")
    parser.add_argument("--reset_trajectory_token", default=False, action="store_true", help="Reset trajectory tokens during finetuning")
    parser.add_argument("--reset_trajectory_decoder", default=False, action="store_true", help="Reset trajectory decoder during finetuning")
    # loss
    parser.add_argument("--loss_action", default=False, action="store_true")
    parser.add_argument("--loss_image", default=False, action="store_true")
    
    # calvin
    parser.add_argument("--except_lang", default=False, action="store_true")
    # gpt2
    parser.add_argument("--transformer_layers", default=12, type=int)
    parser.add_argument("--hidden_dim", default=384, type=int)
    parser.add_argument("--transformer_heads", default=12, type=int)
    # pretrain, finetune, evaluate
    parser.add_argument('--phase', required=True, help='pretrain, finetune, evaluate')
    # libero 
    parser.add_argument("--libero_path", default="/home/yifan26/LIBERO")
    parser.add_argument("--libero_img_size", default=128, type=int)
    parser.add_argument("--libero_eval_max_steps", default=600, type=int)
    parser.add_argument("--gripper_width", default=False, action="store_true")
    parser.add_argument("--load_libero_file", type=str, default="h5")
    parser.add_argument("--eval_libero_ensembling", default=False, action="store_true")
    parser.add_argument("--ensembling_temp", default=0.01, type=float)
    
    # Metaworld evaluation arguments
    parser.add_argument(
        "--metaworld_benchmark", 
        type=str, 
        choices=["MT1", "MT10", "MT50", "ML1", "ML10", "ML45"],
        default="MT1",
        help="Metaworld benchmark to evaluate (MT=Multi-Task, ML=Meta-Learning)"
    )
    parser.add_argument(
        "--metaworld_env_name", 
        type=str, 
        default="reach-v3",
        help="Environment name for MT1/ML1 (e.g., 'reach-v3', 'push-v3', 'pick-place-v3')"
    )
    parser.add_argument(
        "--metaworld_max_steps", 
        type=int, 
        default=500,
        help="Maximum steps per episode for Metaworld evaluation"
    )
    parser.add_argument(
        "--metaworld_num_episodes", 
        type=int, 
        default=50,
        help="Number of episodes to evaluate per task"
    )
    parser.add_argument(
        "--metaworld_action_scale_factor", 
        type=float, 
        default=0.1,
        help="Scale factor for converting Seer absolute position actions to Metaworld delta actions. Lower values = smaller movements per step."
    )
    parser.add_argument(
        "--metaworld_img_size", 
        type=int, 
        default=128,
        help="Image size (width and height) for Metaworld rendering"
    )
    parser.add_argument(
        "--metaworld_camera_name", 
        type=str, 
        default="corner",
        choices=["corner", "topview", "behindGripper", "gripperPOV", "corner2", "corner3"],
        help="Camera viewpoint for Metaworld rendering"
    )
    # real
    parser.add_argument("--real_dataset_names", type=str)
    parser.add_argument("--real_val_dataset_names", type=str, default="real_world_val", 
                       help="Name of validation dataset for real-world data")
    parser.add_argument("--enable_validation", action="store_true", default=False,
                       help="Enable validation during training for real-world data")
    parser.add_argument("--use_aug_data", default=False, action="store_true")
    parser.add_argument("--load_affordance", default=False, action="store_true",
                        help="Load affordance labels (mask/waypoint/spatialpoint/cotracker). "
                             "Disable when not using affordance losses to suppress missing-file warnings.")
    parser.add_argument("--real_eval_max_steps", default=600, type=int)
    # preprocess
    parser.add_argument("--max_rel_pos", type=float, default=0.02)
    parser.add_argument("--max_rel_orn", type=float, default=0.05)
    parser.add_argument("--magic_scaling_factor_pos", type=float, default=1.0)
    parser.add_argument("--magic_scaling_factor_orn", type=float, default=1.0)
    # for eval
    if is_eval:
        parser.add_argument("--calvin_conf_path", type=str, help="path to calvin configuration file")
        parser.add_argument("--future_act_len", default=-1, type=int)
        parser.add_argument(
            "--visualize",
            default=False,
            action="store_true"
        )
        parser.add_argument(
            "--reset",
            default=False,
            action="store_true"
        )
        parser.add_argument(
            "--diverse_inst",
            default=False,
            action="store_true"
        )
        parser.add_argument("--pad_length", type=int, default=-1)
    parser.add_argument("--window_size", type=int, default=13)
    parser.add_argument("--vit_checkpoint_path", type=str)
    
    # Data collection related parameters
    parser.add_argument(
        "--rollout_output_dir",
        type=str,
        default="./libero_10_rollouts",
        help="Directory to save collected trajectory data"
    )
    
    args = parser.parse_args()

    return parser

    # if args.dataloading_type == "seer":
    #     if args.phase == "pretrain":
    #         if args.finetune_type == "calvin":
    #             args.window_size = args.sequence_length + args.future_steps 
    #         else:
    #             args.window_size = args.sequence_length
    #     elif args.phase == "finetune":
    #         args.window_size = args.sequence_length + args.future_steps
