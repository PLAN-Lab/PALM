import os
import random
from functools import partial
from copy import deepcopy
from timm.models.vision_transformer import Block
import torch
import time
from torch import nn
import torch.nn.functional as F
import clip
import numpy as np
from models.vit_mae import MaskedAutoencoderViT
from models.perceiver_resampler import PerceiverResampler
from models.gpt2 import GPT2Model
from transformers import GPT2Config
from pdb import set_trace
import random 


def generate_attention_mask(K, num_A, num_B, atten_goal, atten_goal_state,
                            atten_only_obs,
                            attn_robot_proprio_state,
                            mask_l_obs_ratio,
                            num_obs_token, action_pred_steps):
    # num_A: 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
    # num_A: text, state, image_embedding, image_cls_token_embedding
    # num_B: self.NUM_OBS_TOKEN+self.action_pred_steps
    # num_B: obs_tokens(if exists), action_pred_token, state_pred_token (if exists)
    sequence_length = (num_A + num_B) * K
    attention_mask = torch.zeros((sequence_length, sequence_length))
    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B
        
        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = -float('inf')
        
        # the sub-sub-sequence B can not be attended to
        attention_mask[:, start_index+num_A:end_index] = -float('inf')
        
        # if obs_token exists, action_pred_token should attend to it
        if num_obs_token > 0 and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
        if num_obs_token > 0 and atten_only_obs and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps] = -float('inf')
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+2:start_index+num_A] = 0.0
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
            if attn_robot_proprio_state:
                attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+1:start_index+2] = 0.0
            if mask_l_obs_ratio > 0:
                count = int(mask_l_obs_ratio * (num_obs_token))
                selected_numbers = np.random.choice(range(num_obs_token), size=count, replace=False)
                for num in selected_numbers:
                    attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A+num] = -float('inf')
        if num_obs_token > 0 and atten_goal:
            if i < K - atten_goal:
                pred_end_index = (i + atten_goal) * (num_A + num_B)
                if atten_goal_state:
                    attention_mask[start_index+num_A:start_index+num_A+num_obs_token,pred_end_index+1:pred_end_index+2] = 0.0

    return attention_mask

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

class PALMAgent(nn.Module):
    def __init__(
        self,
        finetune_type,
        clip_device,
        vit_checkpoint_path,
        sequence_length=10,
        num_resampler_query=9,
        num_obs_token_per_image=10,
        obs_pred=False,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=False,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        calvin_input_image_size=224,
        patch_size=16,
        mask_ratio=0.0,
        num_token_per_timestep=41,
        input_self=False,
        action_pred_steps=1,
        transformer_layers=12,
        hidden_dim=384,
        transformer_heads=12,
        phase="",
        gripper_width=False,
        global_pred=False,
        global_pred_steps=1,
        mask_height=None,
        mask_width=None,
        local_pred=False,
        local_pred_steps=1,
        spatial_pred=False,
        spatial_pred_steps=1,
        spatial_num_candidates=10,
        dynamic_pred=False,
        dynamic_pred_steps=1,
        dynamic_patch_size=8,
        no_pred_gripper_dynamic=False,
        no_unshuffle=False,
        flow_as_mask=False,
        # DiT decoder and progress prediction parameters
        use_dit_decoder=False,
        predict_progress=False,
        dit_hidden_size=768,
        dit_depth=12,
        dit_num_heads=12,
        dit_diffusion_steps=100,
        dit_sampling_steps=10,
        dit_noise_schedule='cosine',
    ):
        super().__init__()
        self.finetune_type = finetune_type
        self.device = clip_device
        self.sequence_length = sequence_length
        self.action_pred_steps = action_pred_steps
        self.obs_pred = obs_pred
        self.atten_goal = atten_goal
        self.atten_goal_state = atten_goal_state
        self.atten_only_obs = atten_only_obs
        self.attn_robot_proprio_state = attn_robot_proprio_state
        self.mask_l_obs_ratio = mask_l_obs_ratio
        self.hidden_dim = hidden_dim
        self.phase = phase
        assert self.phase in ["pretrain", "finetune", "evaluate"]
        self.gripper_width = gripper_width
        # Global Affordance (Mask Prediction)
        self.global_pred = global_pred
        self.global_pred_steps = global_pred_steps
        self.mask_height = mask_height
        self.mask_width = mask_width
        # Local Affordance (Contact Prediction)
        self.local_pred = local_pred
        self.local_pred_steps = local_pred_steps
        # Spatial Affordance: paper §3.3 -> output a SET of M candidate
        # placement points, supervised by a Chamfer-style set-matching loss.
        self.spatial_pred = spatial_pred
        self.spatial_pred_steps = spatial_pred_steps
        self.spatial_num_candidates = max(1, int(spatial_num_candidates))
        # Dynamic Affordance (Trajectory Prediction)
        self.dynamic_pred = dynamic_pred
        self.dynamic_pred_steps = dynamic_pred_steps
        self.dynamic_patch_size = dynamic_patch_size
        self.no_pred_gripper_dynamic = no_pred_gripper_dynamic
        self.no_unshuffle = no_unshuffle
        self.flow_as_mask = flow_as_mask
        self.vit_checkpoint_path = vit_checkpoint_path
        
        # DiT decoder and progress prediction parameters
        self.use_dit_decoder = use_dit_decoder
        self.predict_progress = predict_progress
        self.dit_hidden_size = dit_hidden_size
        self.dit_depth = dit_depth
        self.dit_num_heads = dit_num_heads
        self.dit_diffusion_steps = dit_diffusion_steps
        self.dit_sampling_steps = dit_sampling_steps
        self.dit_noise_schedule = dit_noise_schedule

        # text projector
        self.text_projector = nn.Linear(512, self.hidden_dim)        

        # state encoder
        ARM_STATE_FEATURE_DIM = self.hidden_dim 
        GRIPPER_STATE_FEATURE_DIM = self.hidden_dim
        self.arm_state_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.gripper_state_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.state_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim)

        # action encoder
        self.action_pose_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.action_gripper_position_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.action_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim)

        # vision encoder (frozen)
        self.vision_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # resampler
        self.RESAMPLER_hidden_dim = 768  
        self.NUM_RESAMPLER_QUERY = num_resampler_query
        self.perceiver_resampler = PerceiverResampler(dim=self.RESAMPLER_hidden_dim, num_latents=self.NUM_RESAMPLER_QUERY, depth=3)
        self.image_primary_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_primary_projector = nn.Linear(768, self.hidden_dim)
        self.image_wrist_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_wrist_projector = nn.Linear(768, self.hidden_dim)

        # action_pred_token
        if self.action_pred_steps > 0:
            self.action_pred_token = nn.Parameter(torch.zeros(1, 1, self.action_pred_steps, self.hidden_dim))

        # mask_pred_token
        if self.global_pred and self.global_pred_steps > 0:
            self.global_pred_token = nn.Parameter(torch.zeros(1, 1, self.global_pred_steps * 2, self.hidden_dim))

        # contact_pred_token
        if self.local_pred and self.local_pred_steps > 0:
            # 2 tokens per step: one for primary camera, one for wrist camera
            self.local_pred_token = nn.Parameter(torch.zeros(1, 1, self.local_pred_steps * 2, self.hidden_dim))

        # spatial_pred_token
        if self.spatial_pred and self.spatial_pred_steps > 0:
            # 2 tokens per step: one for primary camera, one for wrist camera
            self.spatial_pred_token = nn.Parameter(torch.zeros(1, 1, self.spatial_pred_steps * 2, self.hidden_dim))

        # trajectory prediction token
        # NUM_OBS_TOKEN_PER_TRAJ: number of observation tokens used for trajectory prediction per camera
        # Similar to image prediction, we use obs tokens to guide trajectory prediction
        if self.dynamic_pred and self.dynamic_pred_steps > 0:
            # Determine number of traj tokens based on whether we predict gripper trajectory
            # Each camera (primary/wrist) gets NUM_OBS_TOKEN_PER_IMAGE tokens to encode trajectory features
            self.NUM_OBS_TOKEN_PER_TRAJ = num_obs_token_per_image  # 10 tokens per camera
            if self.no_pred_gripper_dynamic:
                # Only predict primary camera trajectory
                self.NUM_DYNAMIC_TOKEN = self.NUM_OBS_TOKEN_PER_TRAJ * 1  # 10 tokens total
            else:
                # Predict both primary and wrist trajectories  
                self.NUM_DYNAMIC_TOKEN = self.NUM_OBS_TOKEN_PER_TRAJ * 2  # 20 tokens total
            
            # Trajectory prediction token: [1, 1, NUM_TRAJ_TOKEN, hidden_dim]
            self.dynamic_pred_token = nn.Parameter(torch.zeros(1, 1, self.NUM_DYNAMIC_TOKEN, self.hidden_dim))

        # obs_token
        self.NUM_OBS_TOKEN_PER_IMAGE = num_obs_token_per_image
        self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * 2
        if self.obs_pred:
            self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_OBS_TOKEN, self.hidden_dim))
        
        # causal transformer
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Calculate number of tokens for each prediction task
        if self.obs_pred:
            this_num_obs_token = self.NUM_OBS_TOKEN  # 20 tokens for image observation prediction
        else:
            this_num_obs_token = 0

        if self.global_pred and self.global_pred_steps > 0:
            this_num_mask_token = self.global_pred_steps * 2  # 2 tokens (primary + wrist camera)
        else:
            this_num_mask_token = 0

        if self.local_pred and self.local_pred_steps > 0:
            this_num_contact_token = self.local_pred_steps * 2  # 2 tokens (primary + wrist camera)
        else:
            this_num_contact_token = 0

        if self.spatial_pred and self.spatial_pred_steps > 0:
            this_num_spatial_token = self.spatial_pred_steps * 2  # 2 tokens (primary + wrist camera)
        else:
            this_num_spatial_token = 0

        if self.dynamic_pred and self.dynamic_pred_steps > 0:
            this_num_traj_token = self.NUM_DYNAMIC_TOKEN  # NUM_DYNAMIC_TOKEN (10 or 20 depending on no_pred_gripper_dynamic)
        else:
            this_num_traj_token = 0

        # CRITICAL FIX: Calculate total prediction tokens that action should attend to
        # Action prediction needs to see ALL prediction tasks (obs, mask, contact, spatial, trajectory)
        # to leverage their information for better action prediction
        total_pred_tokens_before_action = (
            this_num_obs_token + 
            this_num_mask_token + 
            this_num_contact_token + 
            this_num_spatial_token + 
            this_num_traj_token
        )

        self.attention_mask = nn.Parameter(generate_attention_mask(
                                    K=self.sequence_length, 
                                    num_A=1+1+self.NUM_RESAMPLER_QUERY*2+1*2,  # 132: text(1) + state(1) + image(NUM_RESAMPLER_QUERY*2) + cls_token(2)
                                    num_B=total_pred_tokens_before_action+self.action_pred_steps,  # All prediction tokens + action tokens
                                    atten_goal=self.atten_goal,
                                    atten_goal_state=self.atten_goal_state,
                                    atten_only_obs=self.atten_only_obs,
                                    attn_robot_proprio_state = self.attn_robot_proprio_state,
                                    mask_l_obs_ratio=self.mask_l_obs_ratio,
                                    num_obs_token=total_pred_tokens_before_action,  # FIXED: Pass total prediction tokens instead of just obs_tokens
                                    action_pred_steps=self.action_pred_steps), 
                                    requires_grad=False)
        num_non_learnable_token_per_timestep = 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
        self.transformer_backbone_position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, 1, self.hidden_dim), requires_grad=True)  # TODO How to initialize this embedding
        config = GPT2Config()
        config.hidden_size = self.hidden_dim
        config.n_layer = transformer_layers
        config.vocab_size = 1
        config.n_head = transformer_heads
        self.transformer_backbone = GPT2Model(config)

        # ========== Action Decoder: MLP or DiT ==========
        # Define MLP_hidden_dim outside if-else for recon_state_decoder usage
        MLP_hidden_dim = self.hidden_dim // 2  # 384 -> 192
        
        # Choose between MLP decoder (original) or DiT decoder (paper version)
        if not self.use_dit_decoder:
            # Option 1: MLP Decoder (original implementation)
            # Simple 3-layer MLP that directly predicts actions
            self.action_decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, MLP_hidden_dim),
                nn.ReLU(),
                nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
                nn.ReLU(),
            )
            self.arm_action_decoder = nn.Sequential(
                nn.Linear(MLP_hidden_dim, 6),  # 6-DOF pose
                torch.nn.Tanh(),
            )
            self.gripper_action_decoder = nn.Sequential(
                nn.Linear(MLP_hidden_dim, 1),  # Binary gripper state
                torch.nn.Sigmoid(),
            )
        else:
            # Option 2: DiT Decoder (paper implementation)
            # Diffusion Transformer for action-progress prediction
            from models.action_model.action_model import ActionModel
            
            # Calculate action output dimension
            # Action: 6D pose + 1D gripper = 7D
            # Progress: 1D (optional, currently disabled)
            if self.predict_progress:
                action_output_dim = 7 + 1  # 6D pose + 1D gripper + 1D progress
            else:
                action_output_dim = 7      # 6D pose + 1D gripper (no progress)
            
            # Create DiT action decoder (Paper Table 7 configuration)
            # Hidden Size: 384, Number of Layers: 12, Number of Heads: 12,
            # Sampling Steps: 10, Noise Schedule: Cosine.
            # NOTE: We pass dit_hidden_size/dit_depth/dit_num_heads explicitly so
            # the DiT actually built matches the paper instead of silently using
            # the legacy DiT-B preset (hidden=768).
            self.dit_action_decoder = ActionModel(
                token_size=self.hidden_dim * self.action_pred_steps,  # Conditioning feature dim
                model_type='DiT-B',                           # only used if explicit dims are None
                in_channels=action_output_dim,                # Output dimension (7 or 8)
                future_action_window_size=self.action_pred_steps,  # Multi-step prediction
                past_action_window_size=0,                    # No history actions
                diffusion_steps=self.dit_diffusion_steps,
                noise_schedule=self.dit_noise_schedule,
                hidden_size=self.dit_hidden_size,             # Paper: 384
                depth=self.dit_depth,                         # Paper: 12
                num_heads=self.dit_num_heads,                 # Paper: 12
            )
            
            # Create DDIM sampler for fast inference (10 steps instead of 100)
            self.dit_ddim_sampler = self.dit_action_decoder.create_ddim(
                ddim_step=self.dit_sampling_steps  # DDIM sampling steps (10)
            )
            
            # Note: When using DiT, arm_action_decoder and gripper_action_decoder are not needed
            # DiT directly outputs the full action vector [6D pose + 1D gripper (+ 1D progress)]
        # ================================================

        self.recon_state_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
        ) # not used
        self.recon_arm_state_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 6),
            torch.nn.Tanh(),
        ) # not used
        self.recon_gripper_state_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 1),
            torch.nn.Sigmoid(),
        ) # not used

        self.IMAGE_DECODER_hidden_dim = self.hidden_dim
        self.NUM_MASK_TOKEN = int(calvin_input_image_size**2 / patch_size / patch_size)  # i.e. num_patch
        self.PATCH_SIZE = patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.IMAGE_DECODER_hidden_dim))
        self.image_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.IMAGE_DECODER_hidden_dim)
        self.image_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_IMAGE + self.NUM_MASK_TOKEN, self.IMAGE_DECODER_hidden_dim), requires_grad=False)  # fixed sin-cos embedding #   cls_token is alse passed to the decoder in mae
        self.image_decoder = nn.Sequential(
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            )
        self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_hidden_dim)
        self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_hidden_dim, self.PATCH_SIZE**2 * 3)

        # Global Affordance Decoder (MLP-based, matching paper: <Global> uses lightweight 2-layer MLP)
        if self.global_pred:
            # Global affordance prediction uses MLP decoder (paper: <Global> uses lightweight 2-layer MLP heads)
            # Output: Binary mask (0/1) for future object segmentation
            GLOBAL_DECODER_hidden_dim = self.hidden_dim // 2  # 192
            
            # Shared feature extractor for global affordance prediction (2-layer MLP)
            self.global_decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, GLOBAL_DECODER_hidden_dim),
                nn.ReLU(),
                nn.Linear(GLOBAL_DECODER_hidden_dim, GLOBAL_DECODER_hidden_dim),
                nn.ReLU(),
            )
            
            # Separate prediction heads for primary and wrist cameras
            # Each head outputs H×W logits for binary mask
            self.global_primary_pred = nn.Linear(GLOBAL_DECODER_hidden_dim, self.mask_height * self.mask_width)
            self.global_wrist_pred = nn.Linear(GLOBAL_DECODER_hidden_dim, self.mask_height * self.mask_width)

        # Local Affordance Decoder (Transformer-based, paper §3.3 + Appendix A.1)
        # Per paper: <Local> uses 2-layer Transformer blocks followed by a linear
        # projection to produce a future *contact heatmap* over the image lattice
        # (NOT a single 2D coordinate). Loss is Focal Loss + KL on the heatmaps
        # (Eq. 4-5).
        if self.local_pred:
            self.LOCAL_DECODER_hidden_dim = self.hidden_dim  # 384

            self.local_decoder_projector = nn.Linear(self.hidden_dim, self.LOCAL_DECODER_hidden_dim)

            # 2-layer Transformer block (Table 6: 2 layers, 16 heads).
            self.local_decoder = nn.Sequential(
                Block(self.LOCAL_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                Block(self.LOCAL_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            )
            self.local_decoder_norm = nn.LayerNorm(self.LOCAL_DECODER_hidden_dim)

            # Output heads produce per-pixel logits of size mask_height*mask_width
            # (one heatmap per camera). Sigmoid is applied implicitly via the
            # focal loss / can be applied externally for visualisation.
            assert self.mask_height is not None and self.mask_width is not None, \
                "mask_height/mask_width must be set when local_pred is enabled (used for the contact heatmap)"
            self.local_primary_pred = nn.Linear(self.LOCAL_DECODER_hidden_dim, self.mask_height * self.mask_width)
            self.local_wrist_pred = nn.Linear(self.LOCAL_DECODER_hidden_dim, self.mask_height * self.mask_width)

        # Spatial Decoder (MLP, paper Appendix A.1: lightweight 2-layer MLP)
        # Output: M candidate 2D placement points per camera per step,
        # supervised with a set-matching objective (paper §3.3 Eq. 6).
        if self.spatial_pred:
            SPATIAL_DECODER_hidden_dim = self.hidden_dim // 2  # 192

            self.spatial_decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, SPATIAL_DECODER_hidden_dim),
                nn.ReLU(),
                nn.Linear(SPATIAL_DECODER_hidden_dim, SPATIAL_DECODER_hidden_dim),
                nn.ReLU(),
            )

            # Each head produces M*2 logits which are squashed to [0, 1] with
            # sigmoid (image-normalised coordinates).
            self.spatial_primary_pred = nn.Sequential(
                nn.Linear(SPATIAL_DECODER_hidden_dim, self.spatial_num_candidates * 2),
                nn.Sigmoid(),
            )
            self.spatial_wrist_pred = nn.Sequential(
                nn.Linear(SPATIAL_DECODER_hidden_dim, self.spatial_num_candidates * 2),
                nn.Sigmoid(),
            )

        # Dynamic Affordance Decoder (paper §3.3 + Appendix A.1)
        # Per the paper, the dynamic head outputs a future *dynamic probability
        # map* over the image lattice that highlights pixels likely to belong to
        # the gripper or other moving objects (NOT a per-patch 2D motion vector).
        # We therefore use a MAE-style transformer decoder that outputs a single
        # logit per output token, which is then reshaped to a (28x28) or
        # (14x14) probability grid (depending on pixel_unshuffle settings).
        if self.dynamic_pred:
            self.DYNAMIC_DECODER_hidden_dim = self.hidden_dim  # 384
            self.PATCH_SIZE = patch_size  # 16

            if self.no_unshuffle:
                # 28 x 28 grid = 784 points, matching CoTracker tracked grid.
                self.NUM_DYNAMIC_MASK_TOKEN = 784 * self.dynamic_pred_steps
                self._dynamic_pred_grid_h = 28
                self._dynamic_pred_grid_w = 28
                # Each token predicts 1 logit per pixel (binary moving / not).
                dynamic_pred_out_dim = 1
            else:
                # 14 x 14 patch grid (224/16). Each patch covers
                # (patch_size//dynamic_patch_size)^2 pixels of the dynamic mask.
                num_patches = int(calvin_input_image_size**2 / patch_size / patch_size)
                self.NUM_DYNAMIC_MASK_TOKEN = num_patches * self.dynamic_pred_steps
                self._dynamic_pred_grid_h = int(num_patches**0.5)
                self._dynamic_pred_grid_w = int(num_patches**0.5)
                dynamic_pred_out_dim = (patch_size // self.dynamic_patch_size) ** 2

            self.dynamic_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.DYNAMIC_DECODER_hidden_dim)

            self.dynamic_decoder = nn.Sequential(
                Block(self.DYNAMIC_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                Block(self.DYNAMIC_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            )
            self.dynamic_decoder_norm = nn.LayerNorm(self.DYNAMIC_DECODER_hidden_dim)

            self.dynamic_mask_token = nn.Parameter(torch.zeros(1, 1, self.DYNAMIC_DECODER_hidden_dim))

            self.dynamic_decoder_position_embedding = nn.Parameter(
                torch.zeros(1, self.NUM_OBS_TOKEN_PER_TRAJ + self.NUM_DYNAMIC_MASK_TOKEN, self.DYNAMIC_DECODER_hidden_dim),
                requires_grad=False,
            )

            # Prediction head outputs raw logits; sigmoid is applied externally
            # by the loss (BCE / Focal Loss). This matches the paper's
            # "calibrated per-pixel probabilities".
            self.dynamic_decoder_pred = nn.Linear(self.DYNAMIC_DECODER_hidden_dim, dynamic_pred_out_dim)
            self._dynamic_pred_out_dim = dynamic_pred_out_dim

        # initialize network
        self.initialize_weights()

        # freeze vision encoder
        vit_checkpoint = torch.load(self.vit_checkpoint_path, map_location='cpu')
        msg = self.vision_encoder.load_state_dict(vit_checkpoint['model'], strict=False)

        # # freeze text encoder
        if os.path.exists("checkpoints/clip/ViT-B-32.pt"):
            self.clip_model, self.image_processor = clip.load("checkpoints/clip/ViT-B-32.pt", device=clip_device)
        else:
            self.clip_model, self.image_processor = clip.load("ViT-B/32", device=clip_device)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(self.IMAGE_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_IMAGE**.5), cls_token=False)
        image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(self.IMAGE_DECODER_hidden_dim, int(self.NUM_MASK_TOKEN**.5), cls_token=False)
        image_decoder_position_embedding = np.concatenate((image_decoder_position_embedding_obs, image_decoder_position_embedding_mask), axis=0)
        self.image_decoder_position_embedding.data.copy_(torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.transformer_backbone_position_embedding, std=.02)
        
        # initialize mask prediction token
        if self.global_pred and self.global_pred_steps > 0:
            torch.nn.init.normal_(self.global_pred_token, std=.02)
        
        # initialize contact prediction token
        if self.local_pred and self.local_pred_steps > 0:
            torch.nn.init.normal_(self.local_pred_token, std=.02)
        
        # initialize spatial prediction token
        if self.spatial_pred and self.spatial_pred_steps > 0:
            torch.nn.init.normal_(self.spatial_pred_token, std=.02)
        
        # initialize trajectory prediction components
        if self.dynamic_pred and self.dynamic_pred_steps > 0:
            # Initialize trajectory prediction token
            torch.nn.init.normal_(self.dynamic_pred_token, std=.02)
            
            # Initialize trajectory mask token
            torch.nn.init.normal_(self.dynamic_mask_token, std=.02)
            
            # Initialize trajectory decoder position embedding with sin-cos embedding
            dynamic_decoder_position_embedding_obs = get_2d_sincos_pos_embed(
                self.DYNAMIC_DECODER_hidden_dim, 
                int(self.NUM_OBS_TOKEN_PER_TRAJ**.5), 
                cls_token=False
            )
            if self.no_unshuffle:
                # No spatial downsampling: 28x28 grid = 784 points
                dynamic_decoder_position_embedding_mask = get_2d_sincos_pos_embed(
                    self.DYNAMIC_DECODER_hidden_dim,
                    28,  # sqrt(784) = 28
                    cls_token=False
                )
            else:
                # With pixel_unshuffle: 14x14 grid = 196 patches
                dynamic_decoder_position_embedding_mask = get_2d_sincos_pos_embed(
                    self.DYNAMIC_DECODER_hidden_dim,
                    int((self.NUM_DYNAMIC_MASK_TOKEN / self.dynamic_pred_steps)**0.5),  # sqrt(196) = 14
                    cls_token=False
                )
            dynamic_decoder_position_embedding = np.concatenate(
                (dynamic_decoder_position_embedding_obs, dynamic_decoder_position_embedding_mask), 
                axis=0
            )
            self.dynamic_decoder_position_embedding.data.copy_(
                torch.from_numpy(dynamic_decoder_position_embedding).float().unsqueeze(0)
            )
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm may not have bias if elementwise_affine=False
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _init_model_type(self):
        self.vision_encoder_type = next(self.vision_encoder.parameters()).type()
        self.perceiver_resampler_type = next(self.perceiver_resampler.parameters()).type()
        self.transformer_backbone_type = next(self.transformer_backbone.parameters()).type()
        
        # Initialize action decoder type based on decoder choice
        if not self.use_dit_decoder:
            # MLP decoder type
            self.action_decoder_type = next(self.action_decoder.parameters()).type()
        else:
            # DiT decoder type
            self.dit_action_decoder_type = next(self.dit_action_decoder.net.parameters()).type()

        # Initialize global decoder type if global prediction is enabled
        if self.global_pred:
            self.global_decoder_type = next(self.global_decoder.parameters()).type()
        
        # Initialize local decoder type if local prediction is enabled
        if self.local_pred:
            self.local_decoder_type = next(self.local_decoder.parameters()).type()
        
        # Initialize spatial decoder type if spatial prediction is enabled
        if self.spatial_pred:
            self.spatial_decoder_type = next(self.spatial_decoder.parameters()).type()
        
        # Initialize dynamic decoder type if dynamic prediction is enabled
        if self.dynamic_pred:
            self.dynamic_decoder_type = next(self.dynamic_decoder.parameters()).type()


    def forward(self, image_primary, image_wrist, state, text_token, action=None, progress=None, mask_primary=None, mask_wrist=None, contact_primary=None, contact_wrist=None, spatial_primary=None, spatial_wrist=None, traj_static=None, visibility_static=None, traj_gripper=None, visibility_gripper=None):  
        if self.training and self.phase == "pretrain":
            # Calculate number of tokens for each prediction task
            if self.obs_pred:
                this_num_obs_token = self.NUM_OBS_TOKEN  # 20 tokens for image observation prediction
            else:
                this_num_obs_token = 0
            
            if self.global_pred and self.global_pred_steps > 0:
                this_num_mask_token = self.global_pred_steps * 2  # 2 tokens (primary + wrist camera)
            else:
                this_num_mask_token = 0
            
            if self.local_pred and self.local_pred_steps > 0:
                this_num_contact_token = self.local_pred_steps * 2  # 2 tokens (primary + wrist camera)
            else:
                this_num_contact_token = 0
            
            if self.spatial_pred and self.spatial_pred_steps > 0:
                this_num_spatial_token = self.spatial_pred_steps * 2  # 2 tokens (primary + wrist camera)
            else:
                this_num_spatial_token = 0
            
            if self.dynamic_pred and self.dynamic_pred_steps > 0:
                this_num_traj_token = self.NUM_DYNAMIC_TOKEN  # NUM_DYNAMIC_TOKEN (10 or 20 depending on no_pred_gripper_dynamic)
            else:
                this_num_traj_token = 0
            
            # Action prediction needs to see ALL prediction tasks (obs, mask, contact, spatial, trajectory)
            total_pred_tokens_before_action = (
                this_num_obs_token + 
                this_num_mask_token + 
                this_num_contact_token + 
                this_num_spatial_token + 
                this_num_traj_token
            )
            
            self.attention_mask = nn.Parameter(generate_attention_mask(
                            K=self.sequence_length, 
                            num_A=1+1+self.NUM_RESAMPLER_QUERY*2+1*2,  # 132: text(1) + state(1) + image(NUM_RESAMPLER_QUERY*2) + cls_token(2)
                            num_B=total_pred_tokens_before_action+self.action_pred_steps,  # All prediction tokens + action tokens
                            atten_goal=self.atten_goal,
                            atten_goal_state=self.atten_goal_state,
                            atten_only_obs=self.atten_only_obs,
                            attn_robot_proprio_state = self.attn_robot_proprio_state,
                            mask_l_obs_ratio=self.mask_l_obs_ratio,
                            num_obs_token=total_pred_tokens_before_action,  # FIXED: Pass total prediction tokens instead of just obs_tokens
                            action_pred_steps=self.action_pred_steps).to(self.device), 
                            requires_grad=False)
        B, S, _ = state.shape
        device = image_primary.device
        S_AND_FUTURE = image_primary.shape[1]
        image_pred = None
        arm_pred_action, gripper_pred_action = None, None 
        arm_pred_state, gripper_pred_state = None, None
        loss_arm_action = None
        contact_pred_primary, contact_pred_wrist = None, None
        spatial_pred_primary, spatial_pred_wrist = None, None
        traj_pred_primary, traj_pred_wrist = None, None
        
        # text embedding
        with torch.no_grad():
            text_feature = self.clip_model.encode_text(text_token.flatten(0, 1))
            text_feature = text_feature.type(state.type())
        text_embedding = self.text_projector(text_feature)
        text_embedding = text_embedding.view(B, S, -1, self.hidden_dim) 

        # state embedding
        state = state.flatten(0, 1)
        arm_state_feature = self.arm_state_encoder(state[:, :6])
        if not self.gripper_width: 
            gripper_state_one_hot = torch.nn.functional.one_hot(torch.where(state[:, 6:].flatten() < 1, torch.tensor(0).to(device), torch.tensor(1).to(device)), num_classes=2)
            gripper_state_feature = self.gripper_state_encoder(gripper_state_one_hot.type_as(state))
        else:
            gripper_state_feature = self.gripper_state_encoder(state[:, 6:])
        state_embedding = self.state_projector(torch.cat((arm_state_feature, gripper_state_feature), dim=1))
        state_embedding = state_embedding.view(B, S, -1, self.hidden_dim)


        # image feature
        if image_primary.type() != self.vision_encoder_type: 
            image_primary = image_primary.type(self.vision_encoder_type)
            image_wrist = image_wrist.type(self.vision_encoder_type)

        with torch.no_grad():
            image_primary_feature, _, _ = self.vision_encoder.forward_encoder(image_primary.flatten(0, 1), mask_ratio=0.0)
            image_wrist_feature, _, _ = self.vision_encoder.forward_encoder(image_wrist.flatten(0, 1), mask_ratio=0.0)

        if image_primary_feature.type() != self.perceiver_resampler_type:
            image_primary_feature = image_primary_feature.type(self.perceiver_resampler_type)
            image_wrist_feature = image_wrist_feature.type(self.perceiver_resampler_type)
        image_primary_feature = image_primary_feature.view(B, S_AND_FUTURE, image_primary_feature.shape[-2], image_primary_feature.shape[-1])
        image_wrist_feature = image_wrist_feature.view(B, S_AND_FUTURE, image_wrist_feature.shape[-2], image_wrist_feature.shape[-1])
        image_primary_cls_token = image_primary_feature[:, :, :1, :] 
        image_wrist_cls_token = image_wrist_feature[:, :, :1, :]
        image_primary_feature = image_primary_feature[:, :, 1:, :] 
        image_wrist_feature = image_wrist_feature[:, :, 1:, :]
        label_image_primary_feature = image_primary_feature.clone()
        label_image_wrist_feature = image_wrist_feature.clone()

        # perceiver resampler
        image_primary_feature = self.perceiver_resampler(image_primary_feature.reshape(B*S, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))  # mae vit outputs 196 tokens
        image_wrist_feature = self.perceiver_resampler(image_wrist_feature.reshape(B*S, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))
        image_primary_embedding = self.image_primary_projector(image_primary_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_wrist_embedding = self.image_wrist_projector(image_wrist_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_embedding = torch.cat((image_primary_embedding, image_wrist_embedding), dim=2)
        image_cls_token_primary_embedding = self.cls_token_primary_projector(image_primary_cls_token.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_cls_token_wrist_embedding = self.cls_token_wrist_projector(image_wrist_cls_token.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_cls_token_embedding = torch.cat((image_cls_token_primary_embedding, image_cls_token_wrist_embedding), dim=2)

        # aggregate embeddings and add timestep position encoding
        embeddings = torch.cat((text_embedding, state_embedding, image_embedding, image_cls_token_embedding), dim=2)
        # embeddings.shape = [B, S, num_A, hidden_dim] = [4, 16, 132, 384]
        # num_A = 132 = 1(text) + 1(state) + 128(image: NUM_RESAMPLER_QUERY*2) + 2(cls_token: 1*2)

        pred_token_start_idx = embeddings.shape[2]  # Record start index of prediction tokens, pred_token_start_idx = 132
        transformer_input_list = [embeddings]

        # CRITICAL CHANGE: Reorder tokens - action_pred_token moved to the END
        # New token order: [embeddings] + [obs_tokens] + [mask_tokens] + [contact_tokens] + [spatial_tokens] + [traj_tokens] + [action_token]
        # This allows action_pred_token to attend to ALL prediction tasks (obs, mask, contact, spatial, trajectory)
        
        if self.obs_pred:
            # self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_OBS_TOKEN, self.hidden_dim))
            # self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * 2 = 10 * 2 = 20 tokens (2 cameras × 10 tokens per camera)
            transformer_input_list.append(self.obs_tokens.repeat(B, S, 1, 1))  # [B, S, 20, hidden_dim] = [4, 16, 20, 384]

        if self.global_pred and self.global_pred_steps > 0:
            # self.global_pred_token = nn.Parameter(torch.zeros(1, 1, self.global_pred_steps * 2, self.hidden_dim))
            # mask_pred_steps * 2 = 1 * 2 = 2 tokens (primary + wrist camera)
            transformer_input_list.append(self.global_pred_token.repeat(B, S, 1, 1))  # [B, S, 2, hidden_dim] = [4, 16, 2, 384]

        if self.local_pred and self.local_pred_steps > 0:
            # self.local_pred_token = nn.Parameter(torch.zeros(1, 1, self.local_pred_steps * 2, self.hidden_dim))
            # contact_pred_steps * 2 = 1 * 2 = 2 tokens (primary + wrist camera)
            transformer_input_list.append(self.local_pred_token.repeat(B, S, 1, 1))  # [B, S, 2, hidden_dim] = [4, 16, 2, 384]

        if self.spatial_pred and self.spatial_pred_steps > 0:
            # self.spatial_pred_token = nn.Parameter(torch.zeros(1, 1, self.spatial_pred_steps * 2, self.hidden_dim))
            # spatial_pred_steps * 2 = 1 * 2 = 2 tokens (primary + wrist camera)
            transformer_input_list.append(self.spatial_pred_token.repeat(B, S, 1, 1))  # [B, S, 2, hidden_dim] = [4, 16, 2, 384]

        if self.dynamic_pred and self.dynamic_pred_steps > 0:
            # self.dynamic_pred_token = nn.Parameter(torch.zeros(1, 1, self.NUM_DYNAMIC_TOKEN, self.hidden_dim))
            # NUM_DYNAMIC_TOKEN = 10 (if no_pred_gripper_dynamic) or 20 (predict both cameras)
            transformer_input_list.append(self.dynamic_pred_token.repeat(B, S, 1, 1))  # [B, S, NUM_TRAJ_TOKEN, hidden_dim] = [4, 16, 20, 384]

        # ACTION TOKEN MOVED TO LAST - This is the key change!
        if self.action_pred_steps > 0:
            # self.action_pred_token = nn.Parameter(torch.zeros(1, 1, self.action_pred_steps, self.hidden_dim))
            # action_pred_steps = 1 (or more for multi-step action prediction)
            transformer_input_list.append(self.action_pred_token.repeat(B, S, 1, 1))  # [B, S, action_pred_steps, hidden_dim] = [4, 16, 1, 384]

        transformer_input = torch.cat(transformer_input_list, dim=2)
        transformer_input = transformer_input + self.transformer_backbone_position_embedding.repeat(B, 1, transformer_input.shape[-2], 1)  # Add position embedding
        transformer_input = transformer_input.flatten(1, 2)  # [B, S, 179, hidden_dim] → [B, S*179, hidden_dim] = [4, 2864, 384]
        # Flatten for GPT2: GPT2 expects [batch, sequence, features] format

        # Causal transformer forward pass
        if transformer_input.type() != self.transformer_backbone_type:
            transformer_input = transformer_input.type(self.transformer_backbone_type)
        transformer_input = self.embedding_layer_norm(transformer_input)
        transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=self.attention_mask)  # GPT2 backbone
        transformer_output = transformer_output.view(B, S, -1, self.hidden_dim)  # Restore original structure: [4, 2864, 384] → [4, 16, 179, 384]

        # Extract prediction tokens from transformer output
        # Note: Token extraction order follows the NEW token order (with action moved to last)
        current_idx = pred_token_start_idx  # Start from position 132

        if self.obs_pred:
            obs_pred_feature = transformer_output[:, :, current_idx : current_idx+self.NUM_OBS_TOKEN, :]
            # obs_pred_feature.shape = [B, S, NUM_OBS_TOKEN, hidden_dim] = [4, 16, 20, 384]
            # Extract 20 observation prediction tokens from positions 132-151
            current_idx += self.NUM_OBS_TOKEN  # Move to position 152
            obs_pred_embedding = self.image_decoder_obs_pred_projector(obs_pred_feature.reshape(-1, self.hidden_dim))
            #  self.image_decoder_obs_pred_projector的作用将Transformer输出的观测预测特征投影到图像解码器的输入空间, 因为obs_pred_feature.shape = [4, 16, 20, 384]的特征无法之间给image_decoder去解码, 需要转成[1280, 384]维度  # 1280 = 4×16×20
            obs_pred_embedding = obs_pred_embedding.view(B * S * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_OBS_TOKEN_PER_IMAGE, self.IMAGE_DECODER_hidden_dim)
            # 重塑为 [相机数, 每相机token数, 特征维度] -> NUM_OBS_TOKEN // NUM_OBS_TOKEN_PER_IMAGE = num_cameras (20 // 10 = 2) -> obs_pred_embedding.shape = [128, 10, 384], 代表每个视角单张图片未来预测为每个10个预测token
            mask_tokens = self.mask_token.repeat(B * S * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_MASK_TOKEN, 1) # 创建mask tokens用于图像重建, mask_tokens.shape = [128, 196, 384], 每个样本196个mask token
            image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1) # [128, 10, 384] + [128, 196, 384] → [128, 206, 384] ->  206 = 10个预测token + 196个mask token
            image_decoder_input = image_decoder_input + self.image_decoder_position_embedding

            image_decoder_output = self.image_decoder(image_decoder_input) # self.image_decoder来自两层ViT Block (from timm.models.vision_transformer import Block)
            image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN:, :] # 只提取mask token部分 符合MAE ViT机制: [128, 206, 384] → [128, 196, 384]  # 只要最后196个token
            image_pred_feature = self.image_decoder_norm(image_pred_feature.reshape(-1, self.IMAGE_DECODER_hidden_dim)) # self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_hidden_dim), 归一化和像素预测 [128, 196, 384] → [25088, 384]  # 25088 = 128×196
            image_pred = self.image_decoder_pred(image_pred_feature)  # [25088, 384] → [25088, 768]  # 768 = 16×16×3 （3是RGB通道数, 16x6=256代表每个patch包含256个像素点)    #self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_hidden_dim, self.PATCH_SIZE**2 * 3)
            image_pred = image_pred.view(B * S, self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE, self.NUM_MASK_TOKEN, -1)  # [25088, 768] → [64, 2, 196, 768] -> 64 = 4×16 (batch×time), 2 = 相机数, 196 = patch数, 768 = 像素数
        
        # NOTE: Action token extraction moved to AFTER all other prediction tasks
        # This is because action_pred_token is now at the END of the token sequence
        
        # Global Affordance (Mask Prediction with MLP)
        mask_pred_primary = None
        mask_pred_wrist = None
        if self.global_pred and self.global_pred_steps > 0:
            # Extract mask prediction tokens from transformer output
            # Position: after obs_tokens (if enabled), at current_idx
            mask_pred_feature = transformer_output[:, :, current_idx:current_idx+self.global_pred_steps*2, :]
            # mask_pred_feature.shape = [B, S, mask_pred_steps*2, hidden_dim] = [4, 16, 2, 384]
            current_idx += self.global_pred_steps * 2  # Move to next position
            
            # type conversion for mask decoder
            if mask_pred_feature.type() != self.global_decoder_type:
                mask_pred_feature = mask_pred_feature.type(self.global_decoder_type)
            
            # reshape for processing: [B*S*mask_pred_steps*2, hidden_dim]
            B_orig, S_orig = mask_pred_feature.shape[:2]
            mask_pred_feature = mask_pred_feature.reshape(B_orig * S_orig * self.global_pred_steps * 2, self.hidden_dim)
            # mask_pred_feature.shape = [B*S*mask_pred_steps*2, hidden_dim] = [128, 384]
            
            # pass through shared mask decoder (2-layer MLP)
            mask_pred_feature = self.global_decoder(mask_pred_feature)
            # mask_pred_feature.shape = [B*S*mask_pred_steps*2, 192] = [128, 192]
            
            # reshape back to separate primary and wrist features
            mask_pred_feature = mask_pred_feature.reshape(B_orig * S_orig * self.global_pred_steps, 2, -1)
            # mask_pred_feature.shape = [B*S*mask_pred_steps, 2, 192] = [64, 2, 192]
            
            # separate primary and wrist features
            mask_pred_primary_feature = mask_pred_feature[:, 0, :]  # [64, 192]
            mask_pred_wrist_feature = mask_pred_feature[:, 1, :]    # [64, 192]
            
            # generate mask predictions (H×W logits)
            mask_pred_primary = self.global_primary_pred(mask_pred_primary_feature)
            mask_pred_wrist = self.global_wrist_pred(mask_pred_wrist_feature)
            # mask_pred_primary.shape = [B*S*mask_pred_steps, mask_height*mask_width] = [64, 57600]
            # mask_pred_wrist.shape = [B*S*mask_pred_steps, mask_height*mask_width] = [64, 57600]
            
            # reshape to final output format
            mask_pred_primary = mask_pred_primary.reshape(B_orig, S_orig, self.global_pred_steps, self.mask_height, self.mask_width)
            mask_pred_wrist = mask_pred_wrist.reshape(B_orig, S_orig, self.global_pred_steps, self.mask_height, self.mask_width)
            # mask_pred_primary.shape = [B, S, mask_pred_steps, H, W] = [4, 16, 1, 180, 320]
            # mask_pred_wrist.shape = [B, S, mask_pred_steps, H, W] = [4, 16, 1, 180, 320]
        
        # Local Affordance (Contact heatmap, paper §3.3 Eq. 4)
        # Output: [B, S, local_pred_steps, H, W] heatmap logits (per-pixel
        # contact-likelihood) for each camera. Trained with Focal Loss + KL
        # against a Gaussian heatmap centred on the annotated contact point.
        if self.local_pred and self.local_pred_steps > 0:
            contact_pred_feature = transformer_output[:, :, current_idx:current_idx+self.local_pred_steps*2, :]
            current_idx += self.local_pred_steps * 2

            if contact_pred_feature.type() != self.local_decoder_type:
                contact_pred_feature = contact_pred_feature.type(self.local_decoder_type)

            contact_pred_feature = self.local_decoder_projector(contact_pred_feature)

            B_orig, S_orig = contact_pred_feature.shape[:2]
            contact_pred_feature = contact_pred_feature.view(
                B_orig * S_orig, self.local_pred_steps * 2, self.LOCAL_DECODER_hidden_dim
            )

            contact_pred_feature = self.local_decoder(contact_pred_feature)
            contact_pred_feature = self.local_decoder_norm(contact_pred_feature)

            # Even index = primary camera token, odd index = wrist camera token,
            # interleaved per local_pred_step.
            contact_pred_primary_feature = contact_pred_feature[:, ::2, :]   # [B*S, local_pred_steps, hd]
            contact_pred_wrist_feature = contact_pred_feature[:, 1::2, :]    # [B*S, local_pred_steps, hd]

            contact_pred_primary = self.local_primary_pred(contact_pred_primary_feature)  # [B*S, k, H*W]
            contact_pred_wrist = self.local_wrist_pred(contact_pred_wrist_feature)        # [B*S, k, H*W]

            contact_pred_primary = contact_pred_primary.view(
                B_orig, S_orig, self.local_pred_steps, self.mask_height, self.mask_width
            )
            contact_pred_wrist = contact_pred_wrist.view(
                B_orig, S_orig, self.local_pred_steps, self.mask_height, self.mask_width
            )
        
        # Spatial Affordance (Spatial Prediction with MLP)
        if self.spatial_pred and self.spatial_pred_steps > 0:
            # Extract spatial prediction tokens from transformer output
            # Position: after obs_tokens, mask_tokens, and contact_tokens (if enabled), at current_idx
            spatial_pred_feature = transformer_output[:, :, current_idx:current_idx+self.spatial_pred_steps*2, :]
            # spatial_pred_feature.shape = [B, S, spatial_pred_steps*2, hidden_dim] = [4, 16, 2, 384]
            current_idx += self.spatial_pred_steps * 2  # Move to next position
            
            # type conversion for spatial decoder
            if spatial_pred_feature.type() != self.spatial_decoder_type:
                spatial_pred_feature = spatial_pred_feature.type(self.spatial_decoder_type)
            
            # reshape for processing: [B*S*spatial_pred_steps*2, hidden_dim]
            B_orig, S_orig = spatial_pred_feature.shape[:2]
            spatial_pred_feature = spatial_pred_feature.reshape(B_orig * S_orig * self.spatial_pred_steps * 2, self.hidden_dim)
            # spatial_pred_feature.shape = [B*S*spatial_pred_steps*2, hidden_dim] = [128, 384]
            
            # pass through shared spatial decoder (2-layer MLP)
            spatial_pred_feature = self.spatial_decoder(spatial_pred_feature)
            # spatial_pred_feature.shape = [B*S*spatial_pred_steps*2, 192] = [128, 192]
            
            # reshape back to separate primary and wrist features
            spatial_pred_feature = spatial_pred_feature.reshape(B_orig * S_orig * self.spatial_pred_steps, 2, -1)
            # spatial_pred_feature.shape = [B*S*spatial_pred_steps, 2, 192] = [64, 2, 192]
            
            # separate primary and wrist features
            spatial_pred_primary_feature = spatial_pred_feature[:, 0, :]  # [64, 192]
            spatial_pred_wrist_feature = spatial_pred_feature[:, 1, :]    # [64, 192]
            
            # Each prediction is a SET of M normalised (x,y) candidates.
            spatial_pred_primary = self.spatial_primary_pred(spatial_pred_primary_feature)
            spatial_pred_wrist = self.spatial_wrist_pred(spatial_pred_wrist_feature)
            # [B*S*k, M*2]

            # Reshape to [B, S, k, M, 2]
            spatial_pred_primary = spatial_pred_primary.reshape(
                B_orig, S_orig, self.spatial_pred_steps, self.spatial_num_candidates, 2
            )
            spatial_pred_wrist = spatial_pred_wrist.reshape(
                B_orig, S_orig, self.spatial_pred_steps, self.spatial_num_candidates, 2
            )
        
        # Dynamic Affordance (paper §3.3 Eq. 7)
        # Decoder is a MAE-style 2-layer Transformer + linear head; the linear
        # head outputs per-pixel logits of a future motion-probability map
        # (NOT 2D motion vectors as the older code did).
        if self.dynamic_pred and self.dynamic_pred_steps > 0:
            traj_pred_feature = transformer_output[:, :, current_idx:current_idx+self.NUM_DYNAMIC_TOKEN, :]
            current_idx += self.NUM_DYNAMIC_TOKEN

            if traj_pred_feature.type() != self.dynamic_decoder_type:
                traj_pred_feature = traj_pred_feature.type(self.dynamic_decoder_type)

            traj_pred_feature = self.dynamic_decoder_obs_pred_projector(traj_pred_feature.reshape(-1, self.hidden_dim))

            num_cameras = self.NUM_DYNAMIC_TOKEN // self.NUM_OBS_TOKEN_PER_TRAJ
            traj_pred_embedding = traj_pred_feature.view(
                B * S * num_cameras,
                self.NUM_OBS_TOKEN_PER_TRAJ,
                self.DYNAMIC_DECODER_hidden_dim,
            )

            dynamic_mask_tokens = self.dynamic_mask_token.repeat(
                B * S * num_cameras,
                self.NUM_DYNAMIC_MASK_TOKEN,
                1,
            )

            dynamic_decoder_input = torch.cat((traj_pred_embedding, dynamic_mask_tokens), dim=1)
            dynamic_decoder_input = dynamic_decoder_input + self.dynamic_decoder_position_embedding

            dynamic_decoder_output = self.dynamic_decoder(dynamic_decoder_input)
            traj_pred_feature = dynamic_decoder_output[:, -self.NUM_DYNAMIC_MASK_TOKEN:, :]
            traj_pred_feature = self.dynamic_decoder_norm(
                traj_pred_feature.reshape(-1, self.DYNAMIC_DECODER_hidden_dim)
            )

            # [B*S*num_cameras*NUM_DYNAMIC_MASK_TOKEN, dynamic_pred_out_dim]
            traj_pred = self.dynamic_decoder_pred(traj_pred_feature)

            # [B*S, num_cameras, k, NUM_DYNAMIC_MASK_TOKEN/k, dynamic_pred_out_dim]
            traj_pred = traj_pred.view(
                B * S,
                num_cameras,
                self.dynamic_pred_steps,
                self.NUM_DYNAMIC_MASK_TOKEN // self.dynamic_pred_steps,
                -1,
            )

            if self.no_pred_gripper_dynamic:
                traj_pred_primary = traj_pred.squeeze(1)
                traj_pred_wrist = torch.zeros_like(traj_pred_primary)
            else:
                traj_pred_primary = traj_pred[:, 0, :, :, :]
                traj_pred_wrist = traj_pred[:, 1, :, :, :]
        
        # Action prediction - MOVED TO LAST!
        # Extract action token from the FINAL position (after all other prediction tokens)
        progress_pred = None  # Initialize progress prediction (currently disabled)
        loss_dit_diffusion = None  # Initialize DiT diffusion loss
        
        if self.action_pred_steps > 0:
            # Extract action prediction token from transformer output
            # Position: at the END of the sequence, at current_idx (position 178 if all prediction tasks enabled)
            action_pred_feature = transformer_output[:, :, current_idx:current_idx+self.action_pred_steps, :]
            # action_pred_feature.shape = [B, S, action_pred_steps, hidden_dim] = [4, 16, 1, 384]
            # Extract action tokens from the LAST position (after all affordance prediction tokens)
            
            if not self.use_dit_decoder:
                # ========== MLP Decoder Path (Original) ==========
                # CRITICAL: This action token can attend to ALL previous prediction tokens:
                # - obs_tokens (positions 132-151)
                # - mask_tokens (positions 152-153)
                # - contact_tokens (positions 154-155)
                # - spatial_tokens (positions 156-157)
                # - traj_tokens (positions 158-177)
                # This enables action prediction to leverage information from ALL prediction tasks!
                
                # Pass through MLP action decoder to generate action predictions
                action_pred_feature = self.action_decoder(action_pred_feature)
                # action_pred_feature.shape = [B, S, action_pred_steps, MLP_hidden_dim] = [4, 16, 1, 192]
                
                # Generate arm action (6-DOF: x, y, z, rx, ry, rz)
                arm_pred_action = self.arm_action_decoder(action_pred_feature)
                # arm_pred_action.shape = [B, S, action_pred_steps, 6] = [4, 16, 1, 6]
                
                # Generate gripper action (1-DOF: open/close)
                gripper_pred_action = self.gripper_action_decoder(action_pred_feature)
                # gripper_pred_action.shape = [B, S, action_pred_steps, 1] = [4, 16, 1, 1]
            
                # MLP does not predict progress
                progress_pred = None
            
            else:
                # ========== DiT Decoder Path (Paper §3.4) ==========
                B_orig, S_orig = action_pred_feature.shape[:2]

                if self.training:
                    if action is None:
                        raise ValueError("Ground truth action must be provided during training when using DiT decoder")

                    # When atten_goal > 0 the GT `action` only covers the first
                    # (S - atten_goal) timesteps. Slice the conditioning
                    # likewise so the DiT loss only supervises matching steps.
                    cond_feature = action_pred_feature
                    if action.shape[1] != S_orig:
                        valid_S = action.shape[1]
                        cond_feature = action_pred_feature[:, :valid_S]
                        S_used = valid_S
                    else:
                        S_used = S_orig

                    if self.predict_progress:
                        if progress is not None:
                            target = torch.cat([action, progress], dim=-1)
                        else:
                            # No progress GT this batch -> append zeros so the
                            # diffusion still trains (the supervision for the
                            # progress channel can be zeroed out separately by
                            # the caller; here we only keep the architecture
                            # consistent).
                            zeros = torch.zeros(*action.shape[:-1], 1, device=action.device, dtype=action.dtype)
                            target = torch.cat([action, zeros], dim=-1)
                    else:
                        target = action

                    target_reshaped = target.reshape(B_orig * S_used, self.action_pred_steps, -1)
                    condition = cond_feature.reshape(
                        B_orig * S_used, 1, self.action_pred_steps * self.hidden_dim
                    )

                    loss_dit_diffusion = self.dit_action_decoder.loss(x=target_reshaped, z=condition)

                    arm_pred_action = action[..., :6]
                    gripper_pred_action = action[..., 6:7]
                    progress_pred = progress if self.predict_progress else None

                else:
                    condition = action_pred_feature.reshape(
                        B_orig * S_orig, 1, self.action_pred_steps * self.hidden_dim
                    )

                    sampled_actions = self.dit_ddim_sampler.p_sample_loop(
                        model=self.dit_action_decoder.net,
                        shape=(
                            B_orig * S_orig,
                            self.action_pred_steps,
                            7 + (1 if self.predict_progress else 0),
                        ),
                        clip_denoised=True,
                        model_kwargs={'z': condition},
                    )

                    sampled_actions = sampled_actions.reshape(B_orig, S_orig, self.action_pred_steps, -1)

                    arm_pred_action = sampled_actions[..., :6]
                    gripper_pred_action = sampled_actions[..., 6:7]

                    if self.predict_progress:
                        progress_pred = sampled_actions[..., 7:8]
                    else:
                        progress_pred = None
        
        return arm_pred_action, gripper_pred_action, progress_pred, loss_dit_diffusion, image_pred, arm_pred_state, gripper_pred_state, loss_arm_action, mask_pred_primary, mask_pred_wrist, contact_pred_primary, contact_pred_wrist, spatial_pred_primary, spatial_pred_wrist, traj_pred_primary, traj_pred_wrist
