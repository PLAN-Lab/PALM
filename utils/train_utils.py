import time
from contextlib import suppress

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from einops import rearrange
from pdb import set_trace
import numpy as np
import torch.distributed as dist
import math


# ---------------------------------------------------------------------------
# Affordance loss helpers (paper §3.3)
# ---------------------------------------------------------------------------
def sigmoid_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "mean"):
    r"""Pixel-wise sigmoid focal loss used for binary masks (Lin et al., 2017).

    Paper PALM §3.3 Eq. (3,4): both global and local affordance losses use a
    pixel-wise focal loss term (\mathcal{L}_{FL}).
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * (1.0 - p_t).clamp(min=0.0).pow(gamma)
    if alpha is not None and alpha > 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def soft_dice_loss(logits, targets, eps: float = 1.0):
    r"""Soft-Dice loss on sigmoid logits (paper §3.3 Eq. 3 \mathcal{L}_{Dice}).

    `logits`/`targets` are expected to share the same shape; the spatial dims
    (anything beyond the leading batch dim) are reduced together.
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


def gaussian_heatmap_from_points(points, height: int, width: int, sigma: float = 4.0):
    """Build a Gaussian heatmap centred at each (x, y) normalised point.

    Args:
        points: [..., 2] tensor with values in [0, 1]; (x, y) order.
        height/width: output grid size.
        sigma: Gaussian std in pixels.

    Returns:
        Tensor of shape [..., height, width] with values in [0, 1] (the peak
        pixel is 1 at the point location).
    """
    device = points.device
    dtype = points.dtype
    leading = points.shape[:-1]

    # (x, y) in pixel coords, kept in float
    px = points[..., 0].clamp(0.0, 1.0) * (width - 1)
    py = points[..., 1].clamp(0.0, 1.0) * (height - 1)
    px = px.unsqueeze(-1).unsqueeze(-1)   # [..., 1, 1]
    py = py.unsqueeze(-1).unsqueeze(-1)   # [..., 1, 1]

    yy = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)

    # Broadcast against arbitrary leading dims
    while yy.dim() < (len(leading) + 2):
        yy = yy.unsqueeze(0)
        xx = xx.unsqueeze(0)

    diff_x2 = (xx - px) ** 2
    diff_y2 = (yy - py) ** 2
    return torch.exp(-(diff_x2 + diff_y2) / (2.0 * sigma * sigma))


def heatmap_kl_loss(pred_logits, target_heatmap, eps: float = 1e-6):
    """KL divergence between l1-normalised target heatmap and the softmax-normalised
    prediction (paper §3.3 Eq. 4-5).

    Both `pred_logits` and `target_heatmap` have shape [..., H*W] / [..., H, W].
    """
    pred = pred_logits.flatten(start_dim=1)
    tgt = target_heatmap.flatten(start_dim=1)
    log_pred = F.log_softmax(pred, dim=-1)
    tgt_sum = tgt.sum(dim=-1, keepdim=True).clamp(min=eps)
    tgt_norm = tgt / tgt_sum
    return F.kl_div(log_pred, tgt_norm, reduction="batchmean")


def set_matching_loss(pred_set, target_points):
    """Paper §3.3 Eq. 6: set-matching loss for spatial affordance.

    Args:
        pred_set:      [..., M, 2] M predicted candidates.
        target_points: [..., C, 2] (or [..., 2] for a single GT) target points.
    """
    if target_points.dim() == pred_set.dim() - 1:
        target_points = target_points.unsqueeze(-2)  # add C=1 axis
    # pred_set: [N, M, 2], target_points: [N, C, 2]
    # pairwise distance ||p_c - p_m||^2: [N, C, M]
    diff = target_points.unsqueeze(-2) - pred_set.unsqueeze(-3)
    dist2 = (diff ** 2).sum(dim=-1)
    # For each target c, take min over predictions m -> [N, C]
    min_dist2_per_target, _ = dist2.min(dim=-1)
    return min_dist2_per_target.mean()


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_ckpt_name(args, epoch=-1):
    return f'{epoch}.pth'

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def normalize_patchfied_image(patchfied_imgs):
    mean = patchfied_imgs.mean(dim=-1, keepdim=True)
    var = patchfied_imgs.var(dim=-1, keepdim=True)
    patchfied_imgs = (patchfied_imgs - mean) / (var + 1.e-6)**.5

    return patchfied_imgs

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()
    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # Skip this batch if it's None (happens when all samples have incomplete affordance labels)
        if batch_calvin is None:
            print(f"⚠️  Skipping batch {num_steps}: No valid samples with complete affordance labels")
            continue

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
        
        # Load affordance labels (guaranteed to be present, as filtered by collator)
        masks_primary = batch_calvin[6].to(device_id, dtype=cast_dtype, non_blocking=True)  # [B, T, H, W]
        masks_wrist = batch_calvin[7].to(device_id, dtype=cast_dtype, non_blocking=True)    # [B, T, H, W]
        contacts_primary = batch_calvin[8].to(device_id, dtype=cast_dtype, non_blocking=True)  # [B, T, 2]
        contacts_wrist = batch_calvin[9].to(device_id, dtype=cast_dtype, non_blocking=True)    # [B, T, 2]
        spatials_primary = batch_calvin[10].to(device_id, dtype=cast_dtype, non_blocking=True)  # [B, T, 2]
        spatials_wrist = batch_calvin[11].to(device_id, dtype=cast_dtype, non_blocking=True)    # [B, T, 2]
        trajs_static = batch_calvin[12].to(device_id, dtype=cast_dtype, non_blocking=True)         # [B, T, 784, 2]
        visibility_static = batch_calvin[13].to(device_id, dtype=cast_dtype, non_blocking=True)    # [B, T, 784]
        trajs_gripper = batch_calvin[14].to(device_id, dtype=cast_dtype, non_blocking=True)        # [B, T, 784, 2]
        visibility_gripper = batch_calvin[15].to(device_id, dtype=cast_dtype, non_blocking=True)   # [B, T, 784]
        
        # self_key_point
        self_keypoints = None
        
        # actions
        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # label. [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        input_image_primary = images_primary[:, :args.sequence_length, :]
        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]
        
        # Prepare mask input data for model forward pass
        input_mask_primary = masks_primary[:, :args.sequence_length, :]  # [B, T, H, W]
        input_mask_wrist = masks_wrist[:, :args.sequence_length, :]      # [B, T, H, W]
        
        # Prepare contact point input data for model forward pass
        input_contact_primary = contacts_primary[:, :args.sequence_length, :]  # [B, T, 2]
        input_contact_wrist = contacts_wrist[:, :args.sequence_length, :]      # [B, T, 2]
        
        # Prepare spatial point input data for model forward pass
        input_spatial_primary = spatials_primary[:, :args.sequence_length, :]  # [B, T, 2]
        input_spatial_wrist = spatials_wrist[:, :args.sequence_length, :]      # [B, T, 2]

        # Prepare trajectory input data for model forward pass
        input_traj_static = trajs_static[:, :args.sequence_length, :, :]        # [B, T, 784, 2]
        input_visibility_static = visibility_static[:, :args.sequence_length, :]  # [B, T, 784]
        input_traj_gripper = trajs_gripper[:, :args.sequence_length, :, :]      # [B, T, 784, 2]
        input_visibility_gripper = visibility_gripper[:, :args.sequence_length, :]  # [B, T, 784]

        # label action
        label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2)

        # Synthesise a progress GT when predict_progress=True but real progress
        # labels are not available. We use a monotonic 0->1 ramp over the
        # current window, which is a weak but valid prior under the paper's
        # "monotonic, stage-consistent" assumption (§3.4). When real progress
        # annotations are added to the dataloader (paper Appendix A.2 - manually
        # annotated functional keyframes), simply forward them here instead.
        label_progress = None
        if getattr(args, "predict_progress", False):
            B_p = label_actions.shape[0]
            S_p = label_actions.shape[1]
            k_p = label_actions.shape[2]
            steps = args.sequence_length
            denom = max(steps + k_p - 1, 1)
            base = torch.arange(steps, device=device_id, dtype=cast_dtype) / denom
            # [B, S, k, 1]: position(t) + offset(j) for each predicted step j.
            offsets = torch.arange(k_p, device=device_id, dtype=cast_dtype) / denom
            label_progress = (base[:S_p, None] + offsets[None, :]).clamp(0.0, 1.0)
            label_progress = label_progress.unsqueeze(0).expand(B_p, S_p, k_p).unsqueeze(-1).contiguous()

        with autocast():  # Forward pass
            (arm_pred_action, gripper_pred_action, progress_pred, loss_dit_diffusion, image_pred, arm_pred_state, gripper_pred_state,
             loss_arm_action, mask_pred_primary, mask_pred_wrist, contact_pred_primary, contact_pred_wrist,
             spatial_pred_primary, spatial_pred_wrist, traj_pred_primary, traj_pred_wrist) = model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=label_actions,
                progress=label_progress,
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
        # loss_action
        if args.loss_action and args.action_pred_steps:
            if hasattr(args, 'use_dit_decoder') and args.use_dit_decoder:
                # DiT Decoder: use diffusion loss computed in model forward
                # loss_dit_diffusion is already computed in model.forward() during training
                loss_arm_action = loss_dit_diffusion if loss_dit_diffusion is not None else torch.tensor([0.0]).to(device_id)
                loss_gripper_action = torch.tensor([0.0]).to(device_id)  # Gripper loss included in diffusion loss
            else:
                # MLP Decoder: compute action loss separately
                loss_arm_action = torch.nn.functional.smooth_l1_loss(
                                arm_pred_action[:, :args.sequence_length-args.atten_goal], 
                                label_actions[:, :args.sequence_length-args.atten_goal, :, :6].detach())
                loss_gripper_action = torch.nn.functional.binary_cross_entropy(
                                gripper_pred_action[:, :args.sequence_length-args.atten_goal], 
                                label_actions[:, :args.sequence_length-args.atten_goal, :, 6:].detach())
        else:
            loss_arm_action = torch.tensor([0.0]).to(device_id)
            loss_gripper_action = torch.tensor([0.0]).to(device_id)

        # loss_image 
        if args.loss_image and args.obs_pred:
            label_image_primary = images_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_wrist = images_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_primary = patchify(label_image_primary, patch_size=args.patch_size)
            label_image_wrist = patchify(label_image_wrist, patch_size=args.patch_size)
            label_image_primary = normalize_patchfied_image(label_image_primary)
            label_image_wrist = normalize_patchfied_image(label_image_wrist)
            image_pred = image_pred.reshape(-1, args.sequence_length, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3])
            image_pred = image_pred[:, :args.sequence_length-args.atten_goal]
            image_pred = image_pred.reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
            loss_image = 0.5 * (torch.nn.functional.mse_loss(
                            image_pred[:, 0, :, :], 
                            label_image_primary.detach()) + 
                            torch.nn.functional.mse_loss(
                            image_pred[:, 1, :, :], 
                            label_image_wrist.detach()))
        else:
            loss_image = torch.tensor([0.0]).to(device_id)
        
        # ========================================================
        # Global Affordance loss (paper §3.3 Eq. 3): Focal Loss + Dice
        # ========================================================
        if (args.loss_global and masks_primary is not None and masks_wrist is not None
                and mask_pred_primary is not None and mask_pred_wrist is not None):
            label_mask_primary = masks_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :]
            label_mask_wrist = masks_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :]
            label_mask_primary = label_mask_primary.flatten(0, 1).float()
            label_mask_wrist = label_mask_wrist.flatten(0, 1).float()

            mp = mask_pred_primary[:, :args.sequence_length-args.atten_goal, :, :, :].flatten(0, 2).float()
            mw = mask_pred_wrist[:, :args.sequence_length-args.atten_goal, :, :, :].flatten(0, 2).float()

            loss_global_primary = sigmoid_focal_loss(mp, label_mask_primary) \
                                  + soft_dice_loss(mp, label_mask_primary)
            loss_global_wrist = sigmoid_focal_loss(mw, label_mask_wrist) \
                                + soft_dice_loss(mw, label_mask_wrist)
            loss_global = 0.5 * (loss_global_primary + loss_global_wrist)
        else:
            loss_global = torch.tensor([0.0]).to(device_id)

        # ========================================================
        # Local Affordance loss (paper §3.3 Eq. 4): Focal Loss + KL on heatmaps
        # The model now predicts a per-pixel contact heatmap (logits over an
        # H x W grid). Targets are Gaussian heatmaps centred on the annotated
        # contact point.
        # ========================================================
        if (args.loss_local and contacts_primary is not None and contacts_wrist is not None
                and contact_pred_primary is not None and contact_pred_wrist is not None):
            label_c_primary = contacts_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :]
            label_c_wrist = contacts_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :]

            # Predictions: [B, S_valid, k, H, W] -> flatten time/step dims
            cp = contact_pred_primary[:, :args.sequence_length-args.atten_goal, :, :, :].flatten(0, 2).float()
            cw = contact_pred_wrist[:, :args.sequence_length-args.atten_goal, :, :, :].flatten(0, 2).float()
            B_v_T = cp.shape[0]
            H, W = cp.shape[-2], cp.shape[-1]

            # Build Gaussian-heatmap targets per (B, T) -> [B*T, H, W]
            label_c_primary = label_c_primary.flatten(0, 1).float()  # [B*T, 2]
            label_c_wrist = label_c_wrist.flatten(0, 1).float()
            # Repeat targets along the local_pred_steps axis if needed
            if label_c_primary.shape[0] != B_v_T:
                repeat_factor = B_v_T // label_c_primary.shape[0]
                label_c_primary = label_c_primary.repeat_interleave(repeat_factor, dim=0)
                label_c_wrist = label_c_wrist.repeat_interleave(repeat_factor, dim=0)

            target_hm_primary = gaussian_heatmap_from_points(label_c_primary, H, W).float()
            target_hm_wrist = gaussian_heatmap_from_points(label_c_wrist, H, W).float()

            # Binary target for the FL term: 1 where target heatmap is > 0.5
            target_bin_primary = (target_hm_primary > 0.5).float()
            target_bin_wrist = (target_hm_wrist > 0.5).float()

            loss_local_primary = sigmoid_focal_loss(cp, target_bin_primary) \
                                 + heatmap_kl_loss(cp, target_hm_primary)
            loss_local_wrist = sigmoid_focal_loss(cw, target_bin_wrist) \
                               + heatmap_kl_loss(cw, target_hm_wrist)
            loss_local = 0.5 * (loss_local_primary + loss_local_wrist)
        else:
            loss_local = torch.tensor([0.0]).to(device_id)

        # ========================================================
        # Spatial Affordance loss (paper §3.3 Eq. 6): set-matching
        # The model now predicts M candidate points per (camera, step).
        # ========================================================
        if (args.loss_spatial and spatials_primary is not None and spatials_wrist is not None
                and spatial_pred_primary is not None and spatial_pred_wrist is not None):
            label_s_primary = spatials_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :]
            label_s_wrist = spatials_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :]
            label_s_primary = label_s_primary.flatten(0, 1).float()
            label_s_wrist = label_s_wrist.flatten(0, 1).float()

            # spatial_pred_*: [B, S, k, M, 2]
            sp = spatial_pred_primary[:, :args.sequence_length-args.atten_goal, :, :, :].flatten(0, 2).float()
            sw = spatial_pred_wrist[:, :args.sequence_length-args.atten_goal, :, :, :].flatten(0, 2).float()

            if label_s_primary.shape[0] != sp.shape[0]:
                rep = sp.shape[0] // label_s_primary.shape[0]
                label_s_primary = label_s_primary.repeat_interleave(rep, dim=0)
                label_s_wrist = label_s_wrist.repeat_interleave(rep, dim=0)

            loss_spatial_primary = set_matching_loss(sp, label_s_primary)
            loss_spatial_wrist = set_matching_loss(sw, label_s_wrist)
            loss_spatial = 0.5 * (loss_spatial_primary + loss_spatial_wrist)
        else:
            loss_spatial = torch.tensor([0.0]).to(device_id)

        # ========================================================
        # Dynamic Affordance loss (paper §3.3 Eq. 7): per-pixel
        # motion-probability map. We approximate the latent-variable loss with
        # Focal Loss + Soft Dice on a binary motion mask derived from
        # CoTracker trajectories (a pixel is "moving" if its tracked
        # displacement during the prediction window exceeds a threshold).
        # ========================================================
        if (args.loss_dynamic and args.dynamic_pred and trajs_static is not None
                and trajs_gripper is not None and traj_pred_primary is not None
                and traj_pred_wrist is not None):

            B = images_primary.shape[0]

            # The trajectory tensor may be shorter than
            # `future_steps + sequence_length - atten_goal + dynamic_pred_steps`
            # because the dataset window_size limits how far in the future we
            # can look. Slice up to whatever is available, then derive
            # T_valid from the actual length so prediction and target stay
            # aligned even at the tail of an episode.
            traj_end = args.future_steps + args.sequence_length - args.atten_goal + args.dynamic_pred_steps
            traj_window_p = trajs_static[:, args.future_steps:traj_end, :, :]
            traj_window_g = trajs_gripper[:, args.future_steps:traj_end, :, :]
            traj_T = traj_window_p.shape[1]
            # Need at least (dynamic_pred_steps + 1) frames to form one window.
            if traj_T <= args.dynamic_pred_steps:
                loss_dynamic = torch.tensor([0.0]).to(device_id)
            else:
                # Use unfold to create a sliding window of length (dynamic_pred_steps+1)
                # so that index 0 is the reference frame and index 1..k are predicted frames.
                tw_p = traj_window_p.unfold(1, args.dynamic_pred_steps + 1, 1)  # [B, T_valid, 784, 2, k+1]
                tw_g = traj_window_g.unfold(1, args.dynamic_pred_steps + 1, 1)
                tw_p = tw_p.permute(0, 1, 4, 2, 3).contiguous()  # [B, T_valid, k+1, 784, 2]
                tw_g = tw_g.permute(0, 1, 4, 2, 3).contiguous()

                ref_p = tw_p[:, :, :1, :, :]  # [B, T_valid, 1, 784, 2]
                ref_g = tw_g[:, :, :1, :, :]
                fut_p = tw_p[:, :, 1:, :, :]  # [B, T_valid, k, 784, 2]
                fut_g = tw_g[:, :, 1:, :, :]

                disp_p = (fut_p - ref_p).norm(dim=-1)  # [B, T_valid, k, 784]
                disp_g = (fut_g - ref_g).norm(dim=-1)
                motion_thresh = 0.01  # in normalised image coords (~ 2 px @ 224)
                tgt_dyn_primary = (disp_p > motion_thresh).float().flatten(0, 1)  # [B*T_valid, k, 784]
                tgt_dyn_wrist = (disp_g > motion_thresh).float().flatten(0, 1)

                # Match predicted spatial layout. The model returns
                # traj_pred_primary of shape [B*S, k, num_tokens_per_step, out_dim].
                pred_p = traj_pred_primary.float()
                pred_w = traj_pred_wrist.float()
                S_full = pred_p.shape[0] // B
                # Use the actual unfolded window count as the reference: align
                # predictions to the same number of valid future steps.
                T_valid = tgt_dyn_primary.shape[0] // B
                T_keep = min(T_valid, S_full)
                pred_p = pred_p.reshape(B, S_full, *pred_p.shape[1:])[:, :T_keep].flatten(0, 1)
                pred_w = pred_w.reshape(B, S_full, *pred_w.shape[1:])[:, :T_keep].flatten(0, 1)
                # Truncate target if T_valid > S_full (model can't predict that far).
                if T_keep < T_valid:
                    tgt_dyn_primary = tgt_dyn_primary.reshape(B, T_valid, *tgt_dyn_primary.shape[1:])[:, :T_keep].flatten(0, 1)
                    tgt_dyn_wrist = tgt_dyn_wrist.reshape(B, T_valid, *tgt_dyn_wrist.shape[1:])[:, :T_keep].flatten(0, 1)

                # Reshape to a flat [N, 784] logit tensor regardless of pixel_unshuffle.
                pred_p_logits = pred_p.flatten(start_dim=2)  # [B*T_keep, k, 784]
                pred_w_logits = pred_w.flatten(start_dim=2)

                loss_dyn_primary = sigmoid_focal_loss(pred_p_logits, tgt_dyn_primary) \
                                    + soft_dice_loss(pred_p_logits, tgt_dyn_primary)
                if args.no_pred_gripper_dynamic:
                    loss_dynamic = loss_dyn_primary
                else:
                    loss_dyn_wrist = sigmoid_focal_loss(pred_w_logits, tgt_dyn_wrist) \
                                      + soft_dice_loss(pred_w_logits, tgt_dyn_wrist)
                    loss_dynamic = 0.5 * (loss_dyn_primary + loss_dyn_wrist)
        else:
            loss_dynamic = torch.tensor([0.0]).to(device_id)

        # ========================================================
        # Progress loss (paper §3.4): the DiT already supervises the progress
        # channel via its diffusion loss. When the user *also* wants an
        # explicit MSE term on the (training-time) progress GT to keep the
        # progress signal monotonic, we apply it here. Note: in pure DiT
        # training mode `progress_pred` returned from the model equals the
        # GT we passed in, so we use `label_progress` directly (which is the
        # actual supervision signal).
        # ========================================================
        if args.predict_progress and args.loss_progress and label_progress is not None:
            # Without a sampled progress prediction during training we cannot
            # close the loop here without an extra DDIM forward; we therefore
            # rely on the DiT's diffusion loss to learn progress alongside
            # actions and only log a constant 0 here. Setting
            # --loss_progress without `predict_progress` is a no-op.
            loss_progress = torch.tensor([0.0]).to(device_id)
        else:
            loss_progress = torch.tensor([0.0]).to(device_id)

        # Total loss (paper Eq. 3-7 + DiT loss + optional progress)
        loss_calvin = (args.loss_arm_action_ratio * loss_arm_action +
                       args.loss_gripper_action_ratio * loss_gripper_action +
                       0.1 * loss_image +
                       args.loss_global_ratio * loss_global +
                       args.loss_local_ratio * loss_local +
                       args.loss_spatial_ratio * loss_spatial +
                       args.loss_dynamic_ratio * loss_dynamic +
                       args.loss_progress_ratio * loss_progress)

        # gradient_accumulation_steps
        loss = loss_calvin / args.gradient_accumulation_steps
        loss_arm_action = loss_arm_action / args.gradient_accumulation_steps
        loss_gripper_action = loss_gripper_action / args.gradient_accumulation_steps
        loss_image = loss_image / args.gradient_accumulation_steps
        loss_global = loss_global / args.gradient_accumulation_steps
        loss_local = loss_local / args.gradient_accumulation_steps
        loss_spatial = loss_spatial / args.gradient_accumulation_steps
        loss_dynamic = loss_dynamic / args.gradient_accumulation_steps
        loss_progress = loss_progress / args.gradient_accumulation_steps
        mv_avg_loss.append(loss.item())

        ### backward pass ###
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                        "loss_arm_action": loss_arm_action.item() * args.gradient_accumulation_steps,
                        "loss_gripper_action": loss_gripper_action.item() * args.gradient_accumulation_steps,
                        "loss_image": loss_image.item() * args.gradient_accumulation_steps,
                        "loss_global": loss_global.item() * args.gradient_accumulation_steps,
                        "loss_local": loss_local.item() * args.gradient_accumulation_steps,
                        "loss_spatial": loss_spatial.item() * args.gradient_accumulation_steps,
                        "loss_dynamic": loss_dynamic.item() * args.gradient_accumulation_steps,
                        "loss_progress": loss_progress.item() * args.gradient_accumulation_steps,
                        "global_step": global_step,
                    },
                )

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({
            "avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon,
            "loss": loss_calvin.item(),
            "loss_image": loss_image.item(),
            "loss_arm_action": loss_arm_action.item(),
            "loss_gripper_action": loss_gripper_action.item(),
            "loss_global": loss_global.item(),
            "loss_local": loss_local.item(),
            "loss_spatial": loss_spatial.item(),
            "loss_dynamic": loss_dynamic.item(),
            "loss_progress": loss_progress.item(),
        })

        # ========== Checkpoint saving: Step-based (every save_every_iter steps) ==========
        # Note: Epoch-based saving is also enabled in train.py (saves at end of each epoch as sft_{epoch}.pth)
        
        # ACTIVE: Step-based checkpoint saving
        if args.save_every_iter != -1 and args.save_checkpoint and global_step % args.save_every_iter == 0 and global_step > 0:
                
            if args.rank == 0:
                import os
                import glob
                
                # Create output directory if it doesn't exist
                checkpoint_dir = args.save_checkpoint_path
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
        
                # Save current checkpoint
                checkpoint_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": get_checkpoint(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }
        
                ckpt_name = f"checkpoint_step_{global_step}.pth"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                print(f"Saving step-based checkpoint to {ckpt_path}", flush=True)
                torch.save(checkpoint_dict, ckpt_path)
                
                # Keep only the last 3 step-based checkpoints to save disk space
                # Find all step-based checkpoint files
                checkpoint_files = sorted(
                    glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth")),
                    key=lambda x: int(x.split('_')[-1].split('.')[0])  # Sort by step number
                )
                
                # Remove old checkpoints if we have more than 3
                if len(checkpoint_files) > 3:
                    for old_ckpt in checkpoint_files[:-3]:  # Keep only last 3
                        print(f"Removing old step-based checkpoint: {old_ckpt}", flush=True)
                        os.remove(old_ckpt)
        
        # COMMENTED OUT: Original step-based checkpoint saving code (kept for reference)
        # if args.save_every_iter != -1 and args.save_checkpoint and global_step % args.save_every_iter == 0 and global_step > 0:
        #         
        #     if args.rank == 0:
        #         import os
        #         import glob
        #         
        #         # Create output directory if it doesn't exist
        #         checkpoint_dir = args.save_checkpoint_path
        #         if not os.path.exists(checkpoint_dir):
        #             os.makedirs(checkpoint_dir)
        # 
        #         # Save current checkpoint
        #         checkpoint_dict = {
        #             "epoch": epoch,
        #             "global_step": global_step,
        #             "model_state_dict": get_checkpoint(model),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #         }
        # 
        #         ckpt_name = f"checkpoint-{global_step}.pt"
        #         ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        #         print(f"Saving checkpoint to {ckpt_path}", flush=True)
        #         torch.save(checkpoint_dict, ckpt_path)
        #         
        #         # Keep only the last 3 checkpoints
        #         # Find all checkpoint files
        #         checkpoint_files = sorted(
        #             glob.glob(os.path.join(checkpoint_dir, "checkpoint-*.pt")),
        #             key=lambda x: int(x.split('-')[-1].split('.')[0])  # Sort by step number
        #         )
        #         
        #         # Remove old checkpoints if we have more than 3
        #         if len(checkpoint_files) > 3:
        #             for old_ckpt in checkpoint_files[:-3]:  # Keep only last 3
        #                 print(f"Removing old checkpoint: {old_ckpt}", flush=True)
        #                 os.remove(old_ckpt)




        #     if args.rank == 0:
        #         import os
        #         if not os.path.exists(f"{args.save_checkpoint_path}/exp/{args.run_name}"):
        #             os.makedirs(f"{args.save_checkpoint_path}/exp/{args.run_name}")

        #         checkpoint_dict = {
        #             "epoch": epoch,
        #             "model_state_dict": get_checkpoint(model),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #         }

        #         ckpt_name = get_ckpt_name(args, global_step)
        #         ckpt_path = os.path.join(f"{args.save_checkpoint_path}/exp", args.run_name, ckpt_name)
        #         print(f"Saving checkpoint to {ckpt_path}")
        #         torch.save(checkpoint_dict, ckpt_path)
        #         if args.delete_previous_checkpoint:
        #             if epoch > 0:
        #                 os.remove(ckpt_path)



def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def get_checkpoint_all_param(model):
    state_dict = model.state_dict()

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
