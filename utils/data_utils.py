import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
        "mask_obs": ["mask_static", "mask_gripper"],
        "contact_obs": ["contact_primary", "contact_wrist"],
        "spatial_obs": ["spatial_primary", "spatial_wrist"],
        "traj_obs": ["traj_static", "traj_gripper", "visibility_static", "visibility_gripper"],
    }
)

# LIBERO-specific obs_config without auxiliary data (mask, contact, spatial, traj)
# Used for libero_10_converted and libero_90_converted datasets which don't have these fields
obs_config_libero = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_scene_obs": 24,
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] =  R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }

def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}

def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}


def process_mask(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Process mask observations from episode data.
    
    Args:
        episode: Episode data containing mask observations
        observation_space: Configuration specifying mask observation keys
        transforms: Transforms to apply (currently unused for masks)
        seq_idx: Starting sequence index (for episode loader)
        window_size: Window size (for episode loader)
        
    Returns:
        Dict containing processed mask observations with shape [T, H, W]
    """
    mask_obs_keys = observation_space["mask_obs"]
    seq_mask_obs_dict = {}
    
    for _, mask_obs_key in enumerate(mask_obs_keys):
        mask_obs = episode[mask_obs_key]
        # expand dims for single environment obs
        if len(mask_obs.shape) != 3:  # Should be [T, H, W]
            mask_obs = np.expand_dims(mask_obs, axis=0)
        assert len(mask_obs.shape) == 3  # [T, H, W]
        
        if window_size == 0 and seq_idx == 0:  # single file loader
            seq_mask_obs_ = torch.from_numpy(mask_obs).float()
        else:  # episode loader
            seq_mask_obs_ = torch.from_numpy(
                mask_obs[seq_idx : seq_idx + window_size]
            ).float()
        
        # Apply transforms if specified (currently no mask transforms)
        if mask_obs_key in transforms:
            seq_mask_obs_ = transforms[mask_obs_key](seq_mask_obs_)
            
        seq_mask_obs_dict[mask_obs_key] = seq_mask_obs_
    
    # shape: N_mask_obs x [T, H, W]
    return {"mask_obs": seq_mask_obs_dict}


def process_contact(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Process contact point observations from episode data.
    
    Args:
        episode: Episode data containing contact point observations
        observation_space: Configuration specifying contact observation keys
        transforms: Transforms to apply (currently unused for contact points)
        seq_idx: Starting sequence index (for episode loader)
        window_size: Window size (for episode loader)
        
    Returns:
        Dict containing processed contact observations with shape [T, 2]
        Each contact point is represented as [x, y] normalized coordinates
    """
    contact_obs_keys = observation_space["contact_obs"]
    seq_contact_obs_dict = {}
    
    for _, contact_obs_key in enumerate(contact_obs_keys):
        contact_obs = episode[contact_obs_key]
        # expand dims for single environment obs
        if len(contact_obs.shape) != 2:  # Should be [T, 2] for (x, y) coordinates
            contact_obs = np.expand_dims(contact_obs, axis=0)
        assert len(contact_obs.shape) == 2  # [T, 2]
        assert contact_obs.shape[1] == 2  # Each contact point has x, y coordinates
        
        if window_size == 0 and seq_idx == 0:  # single file loader
            seq_contact_obs_ = torch.from_numpy(contact_obs).float()
        else:  # episode loader
            seq_contact_obs_ = torch.from_numpy(
                contact_obs[seq_idx : seq_idx + window_size]
            ).float()
        
        # Apply transforms if specified (currently no contact transforms)
        if contact_obs_key in transforms:
            seq_contact_obs_ = transforms[contact_obs_key](seq_contact_obs_)
            
        seq_contact_obs_dict[contact_obs_key] = seq_contact_obs_
    
    # shape: N_contact_obs x [T, 2]
    return {"contact_obs": seq_contact_obs_dict}


def process_spatial(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Process spatial point observations from episode data.
    
    Args:
        episode: Episode data containing spatial point observations
        observation_space: Configuration specifying spatial observation keys
        transforms: Transforms to apply (currently unused for spatial points)
        seq_idx: Starting sequence index (for episode loader)
        window_size: Window size (for episode loader)
        
    Returns:
        Dict containing processed spatial observations with shape [T, 2]
        Each spatial point is represented as [x, y] normalized coordinates
    """
    spatial_obs_keys = observation_space["spatial_obs"]
    seq_spatial_obs_dict = {}
    
    for _, spatial_obs_key in enumerate(spatial_obs_keys):
        spatial_obs = episode[spatial_obs_key]
        # expand dims for single environment obs
        if len(spatial_obs.shape) != 2:  # Should be [T, 2] for (x, y) coordinates
            spatial_obs = np.expand_dims(spatial_obs, axis=0)
        assert len(spatial_obs.shape) == 2  # [T, 2]
        assert spatial_obs.shape[1] == 2  # Each spatial point has x, y coordinates
        
        if window_size == 0 and seq_idx == 0:  # single file loader
            seq_spatial_obs_ = torch.from_numpy(spatial_obs).float()
        else:  # episode loader
            seq_spatial_obs_ = torch.from_numpy(
                spatial_obs[seq_idx : seq_idx + window_size]
            ).float()
        
        # Apply transforms if specified (currently no spatial transforms)
        if spatial_obs_key in transforms:
            seq_spatial_obs_ = transforms[spatial_obs_key](seq_spatial_obs_)
            
        seq_spatial_obs_dict[spatial_obs_key] = seq_spatial_obs_
    
    # shape: N_spatial_obs x [T, 2]
    return {"spatial_obs": seq_spatial_obs_dict}


def process_traj(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    traj_obs_keys = observation_space["traj_obs"]
    seq_traj_obs_dict = {}
    
    # Process primary camera trajectory data
    # traj_static: tracking points from primary camera
    traj_static = episode['traj_static']  # Input shape: [T, 784, 2]
    visibility_static = episode['visibility_static']  # Input shape: [T, 784]
    
    # Validate primary camera data shapes
    if len(traj_static.shape) != 3:
        traj_static = np.expand_dims(traj_static, axis=0)
    if len(visibility_static.shape) != 2:
        visibility_static = np.expand_dims(visibility_static, axis=0)
    
    assert len(traj_static.shape) == 3, f"Expected traj_static shape [T, N, 2], got {traj_static.shape}"
    assert traj_static.shape[2] == 2, f"Expected 2 coordinates per point, got {traj_static.shape[2]}"
    assert len(visibility_static.shape) == 2, f"Expected visibility_static shape [T, N], got {visibility_static.shape}"
    
    # Slice sequence if needed
    if window_size == 0 and seq_idx == 0:  # single file loader - use all frames
        traj_static_tensor = torch.from_numpy(traj_static).float()
        visibility_static_tensor = torch.from_numpy(visibility_static).float()
    else:  # episode loader - slice the sequence
        traj_static_tensor = torch.from_numpy(
            traj_static[seq_idx : seq_idx + window_size]
        ).float()
        visibility_static_tensor = torch.from_numpy(
            visibility_static[seq_idx : seq_idx + window_size]
        ).float()
    
    # Process wrist camera trajectory data
    # traj_gripper: tracking points from wrist camera
    traj_gripper = episode['traj_gripper']  # Input shape: [T, 784, 2]
    visibility_gripper = episode['visibility_gripper']  # Input shape: [T, 784]
    
    # Validate wrist camera data shapes
    if len(traj_gripper.shape) != 3:
        traj_gripper = np.expand_dims(traj_gripper, axis=0)
    if len(visibility_gripper.shape) != 2:
        visibility_gripper = np.expand_dims(visibility_gripper, axis=0)
    
    assert len(traj_gripper.shape) == 3, f"Expected traj_gripper shape [T, N, 2], got {traj_gripper.shape}"
    assert traj_gripper.shape[2] == 2, f"Expected 2 coordinates per point, got {traj_gripper.shape[2]}"
    assert len(visibility_gripper.shape) == 2, f"Expected visibility_gripper shape [T, N], got {visibility_gripper.shape}"
    
    # Slice sequence if needed
    if window_size == 0 and seq_idx == 0:  # single file loader - use all frames
        traj_gripper_tensor = torch.from_numpy(traj_gripper).float()
        visibility_gripper_tensor = torch.from_numpy(visibility_gripper).float()
    else:  # episode loader - slice the sequence
        traj_gripper_tensor = torch.from_numpy(
            traj_gripper[seq_idx : seq_idx + window_size]
        ).float()
        visibility_gripper_tensor = torch.from_numpy(
            visibility_gripper[seq_idx : seq_idx + window_size]
        ).float()
    
    # Apply transforms if specified (currently no trajectory transforms)
    if "traj_static" in transforms:
        traj_static_tensor = transforms["traj_static"](traj_static_tensor)
    if "visibility_static" in transforms:
        visibility_static_tensor = transforms["visibility_static"](visibility_static_tensor)
    if "traj_gripper" in transforms:
        traj_gripper_tensor = transforms["traj_gripper"](traj_gripper_tensor)
    if "visibility_gripper" in transforms:
        visibility_gripper_tensor = transforms["visibility_gripper"](visibility_gripper_tensor)
    
    # Store in dictionary with consistent naming
    seq_traj_obs_dict["traj_static"] = traj_static_tensor  # [T, 784, 2]
    seq_traj_obs_dict["visibility_static"] = visibility_static_tensor  # [T, 784]
    seq_traj_obs_dict["traj_gripper"] = traj_gripper_tensor  # [T, 784, 2]
    seq_traj_obs_dict["visibility_gripper"] = visibility_gripper_tensor  # [T, 784]
    
    # Return format: {"traj_obs": {key: tensor, ...}}
    # Output shapes:
    #   - traj_static: [N, T, 784, 2] - primary camera tracking coordinates
    #   - visibility_static: [N, T, 784] - primary camera visibility flags
    #   - traj_gripper: [N, T, 784, 2] - wrist camera tracking coordinates  
    #   - visibility_gripper: [N, T, 784] - wrist camera visibility flags
    return {"traj_obs": seq_traj_obs_dict}


def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}

def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

def load_partial_traj_data():
    with open('utils/partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data

def subtract_ranges(rangeA, rangeB):
    def subtract_single_range(a, b):
        result = []
        a_start, a_end = a
        b_start, b_end = b

        if b_start > a_end or b_end < a_start:
            # No overlap
            return [a]
        if b_start > a_start:
            result.append((a_start, min(a_end, b_start - 1)))
        if b_end < a_end:
            result.append((max(a_start, b_end + 1), a_end))

        return [r for r in result if r[0] <= r[1]]

    result = rangeA
    for b in rangeB:
        new_result = []
        for a in result:
            new_result.extend(subtract_single_range(a, b))
        result = new_result

    return result

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        *args: Any,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1,
        data_in_ceph=False,
        **kwargs: Any,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.except_lang = key == "except_lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
        self.data_in_ceph = data_in_ceph
        if self.data_in_ceph:
            self.conf_path = '~/petreloss.conf'
            self.client = Client(self.conf_path)
       
        with open('./utils/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        if self.data_in_ceph:
            assert (
                "validation" in self.abs_datasets_dir
                or "training" in self.abs_datasets_dir
            )
            self.validation = "validation" in self.abs_datasets_dir
        else:
            assert (
                "validation" in self.abs_datasets_dir.as_posix()
                or "training" in self.abs_datasets_dir.as_posix()
            )
            self.validation = "validation" in self.abs_datasets_dir.as_posix()
        print(f"loading dataset at {self.abs_datasets_dir}")
        print("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]} if with_lang else {"lang": 'no lang'}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                print(
                    f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx
        
        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)
        
        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )

        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = self.load_pkl
        elif self.save_format == "npz":
            self.load_file = partial(self.load_npz, data_in_ceph=self.data_in_ceph)
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task
            ) = self._build_file_indices_lang()
        elif self.except_lang:
            self.episode_lookup = self._build_file_indices_except_lang()
        else:
            self.episode_lookup = self._build_file_indices()

        if self.data_in_ceph:
            self.naming_pattern, self.n_digits = self.ceph_lookup_naming_pattern()
        else:
            self.naming_pattern, self.n_digits = lookup_naming_pattern(
                self.abs_datasets_dir, self.save_format
            )
    
    def ceph_lookup_naming_pattern(self):
        filenames = self.client.list(self.abs_datasets_dir)
        for filename in filenames:
            if self.save_format in filename:
                break
        filename = self.abs_datasets_dir + f"/{filename}"
        suffix = "." + self.save_format
        stem_suffix = filename.split('/')[-1]
        stem = stem_suffix.replace(suffix, "")
        aux_naming_pattern = re.split(r"\d+", stem)
        naming_pattern = (filename.replace(stem_suffix, aux_naming_pattern[0]), suffix)
        n_digits = len(re.findall(r"\d+", stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        if self.data_in_ceph:
            return f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        else:
            return Path(
                f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
            )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, # abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        try:
            if self.data_in_ceph:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir +f"/{self.lang_folder}/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/{self.lang_folder}/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                    allow_pickle=True,
                ).item()
        except Exception:
            if self.data_in_ceph:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir + "/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
                ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if [start_idx, end_idx] not in partial_st_ed_list:
                    continue
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def _build_file_indices_except_lang(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        lang_data = np.load(
            abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            allow_pickle=True,
        ).item()
        lang_ep_start_end_ids = lang_data["info"]["indx"]

        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        ep_start_end_ids = np.load(abs_datasets_dir / "except_lang_idx" / "except_lang_idx.npy").tolist()

        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
        
        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs

    def load_pkl(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def load_npz(self, filename, data_in_ceph=False):
        if not data_in_ceph:
            return np.load(filename.as_posix())
        else:
            data_bytes = self.client.get(filename, enable_cache=True)
            data = io.BytesIO(data_bytes)
            try:
                data = np.load(data, allow_pickle=True)
            except:
                data = np.load(data)
            return data

def get_calvin_dataset(args, image_processor, tokenizer, epoch=0, floor=False, except_lang=False):
    dataset_path = args.calvin_dataset
    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/training"
    else:
        datasets_dir = Path(dataset_path) / "training"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph,
        key='except_lang' if except_lang else 'lang',
        obs_space=obs_config_libero, 
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  #
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)

def get_calvin_val_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/validation"
    else:
        datasets_dir = Path(dataset_path) / "validation"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph,
        obs_space=obs_config_libero, 
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        shuffle=False,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)

class BaseDroidDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16, # 时间步数量
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        dataset_info: str = "droid_success",
        small_size: int = 0,
        data_in_ceph: bool = False, 
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        magic_scaling_factor_pos: float = 1.0,
        magic_scaling_factor_orn: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_name = dataset_name 
        self.dataset_info = dataset_info
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        if data_in_ceph:
            self.client = Client(self.conf_path)
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        self.data_in_ceph = data_in_ceph
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.magic_scaling_factor_pos = magic_scaling_factor_pos
        self.magic_scaling_factor_orn = magic_scaling_factor_orn
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        if self.data_in_ceph:
            assert self.client.isdir(self.dataset_path)
        else:
            os.path.exists(self.dataset_path)
        assert os.path.exists(f"data_info/{self.dataset_info}.json")
        with open(f"data_info/{self.dataset_info}.json", 'r') as f: #TODO
            self.episode_info_list = json.load(f)
            
            # Limit to first 500 episodes for droid_language_v2 (only these have mask annotations)
            if self.dataset_info == "droid_language_v2":
                max_episodes = 500
                if len(self.episode_info_list) > max_episodes:
                    print(f"INFO: Limiting {self.dataset_info} to first {max_episodes} episodes (have mask annotations)")
                    print(f"      Original: {len(self.episode_info_list)} episodes, Using: {max_episodes} episodes")
                    self.episode_info_list = self.episode_info_list[:max_episodes]
            
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_

        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        seq.update(
            {
                "mask_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["mask_obs"].items()
                }
            }
        )
        seq.update(
            {
                "contact_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["contact_obs"].items()
                }
            }
        )
        seq.update(
            {
                "spatial_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["spatial_obs"].items()
                }
            }
        )
        seq.update(
            {
                "traj_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["traj_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int): 
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        episode_id = self.episode_list[episode_id]
        episodes = []
        for step_id in range(start_id, end_id):
            data_dict = {}
            try:
                step_id = str(step_id).zfill(4)
                other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/other.h5"
                if self.data_in_ceph:
                    other_bytes = self.client.get(other_path, enable_cache=True)
                    other_h5_file = h5py.File(io.BytesIO(other_bytes))
                else:
                    other_h5_file = h5py.File(other_path)
            except:
                print("episode_id :", episode_id)
                print("step_id :", step_id)
                print("num_step_per_episode :", num_step_per_episode)
                print("other_path", f"{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/other.h5")
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id, step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, step_id)
            data_dict["rel_actions"] = self.load_action(other_h5_file)
            data_dict["robot_obs"] = self.load_robot_obs(other_h5_file)
            data_dict["scene_obs"] = self.load_scene_obs(episode_id, step_id)
            
            # Only load mask/contact/spatial/trajectory data for datasets that have them
            # Currently only droid_language_v2 (first 500 episodes) has these annotations
            if self.dataset_info == "droid_language_v2":
                # Load mask data for both cameras
                data_dict["mask_static"] = self.load_mask_primary(episode_id, step_id)
                data_dict["mask_gripper"] = self.load_mask_wrist(episode_id, step_id)
                # Load contact point data for both cameras
                data_dict["contact_primary"] = self.load_contact_primary(episode_id, step_id)
                data_dict["contact_wrist"] = self.load_contact_wrist(episode_id, step_id)
                # Load spatial point data for both cameras
                data_dict["spatial_primary"] = self.load_spatial_primary(episode_id, step_id)
                data_dict["spatial_wrist"] = self.load_spatial_wrist(episode_id, step_id)
                # Load trajectory data for both cameras
                traj_static_tracks, traj_static_visibility = self.load_traj_static(episode_id, step_id)
                traj_gripper_tracks, traj_gripper_visibility = self.load_traj_gripper(episode_id, step_id)
                data_dict["traj_static"] = traj_static_tracks  # shape: (784, 2)
                data_dict["visibility_static"] = traj_static_visibility  # shape: (784,)
                data_dict["traj_gripper"] = traj_gripper_tracks  # shape: (784, 2)
                data_dict["visibility_gripper"] = traj_gripper_visibility  # shape: (784,)
            else:
                # For other datasets, create dummy/zero data with appropriate shapes
                # Mask shapes based on image sizes
                data_dict["mask_static"] = np.zeros((self.image_primary_size, self.image_primary_size), dtype=np.uint8)
                data_dict["mask_gripper"] = np.zeros((self.image_wrist_size, self.image_wrist_size), dtype=np.uint8)
                # Contact and spatial points: (2,) for x,y coordinates
                data_dict["contact_primary"] = np.zeros(2, dtype=np.float32)
                data_dict["contact_wrist"] = np.zeros(2, dtype=np.float32)
                data_dict["spatial_primary"] = np.zeros(2, dtype=np.float32)
                data_dict["spatial_wrist"] = np.zeros(2, dtype=np.float32)
                # Trajectory data: 784 points (28x28 grid) with (x,y) coordinates
                data_dict["traj_static"] = np.zeros((784, 2), dtype=np.float32)
                data_dict["visibility_static"] = np.zeros(784, dtype=np.float32)
                data_dict["traj_gripper"] = np.zeros((784, 2), dtype=np.float32)
                data_dict["visibility_gripper"] = np.zeros(784, dtype=np.float32)
            episodes.append(data_dict)
   
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        # rgb_static: [16, H, W, 3] # image_primary
        # rgb_gripper: [16, H, W, 3] # image_wrist
        # rel_actions: [16, 7]
        # robot_obs: [16, 8]
        # mask_static: [16, H, W] # primary camera mask (all zeros)
        # mask_gripper: [16, H, W] # wrist camera mask (binary)
        episode["language"] = self.load_language_instruction(other_h5_file, self.language_mode)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_mask_obs = process_mask(episode, self.observation_space, self.transforms)
        seq_contact_obs = process_contact(episode, self.observation_space, self.transforms)
        seq_spatial_obs = process_spatial(episode, self.observation_space, self.transforms)
        seq_traj_obs = process_traj(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_mask_obs,
            **seq_contact_obs,
            **seq_spatial_obs,
            **seq_traj_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["droid_episode_id"] = episode_id

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        if self.data_in_ceph:
            image_primary = self.client.get(image_primary, enable_cache=True)
            image_primary = io.BytesIO(image_primary)
        image_primary = np.array(Image.open(image_primary).convert("RGB"))

        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg'
        if self.data_in_ceph:
            image_wrist = self.client.get(image_wrist, enable_cache=True)
            image_wrist = io.BytesIO(image_wrist)
        image_wrist = np.array(Image.open(image_wrist).convert("RGB")) 

        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_h5_file, language_mode="language_instruction"):
        if "full" not in self.dataset_info:
            language_instruction = other_h5_file[language_mode][()].decode('utf-8')
        else:
            language_instruction = "No language instruction."

        return language_instruction
        
    def load_action(self, other_h5_file, max_rel_pos=0.02, max_rel_orn=0.05, magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        action = other_h5_file["action_delta_wrist_pose"][()]
        action[:3] /= (self.max_rel_pos * self.magic_scaling_factor_pos)
        action[3:6] /= (self.max_rel_orn * self.magic_scaling_factor_orn)

        return action

    def load_robot_obs(self, other_h5_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        robot_obs[:6] = other_h5_file['observation']['gripper_pose6d'][()]
        robot_obs[-1] = other_h5_file['observation']['gripper_open_state'][()]
        robot_obs[7:14] = other_h5_file['observation']['joint_position'][()]

        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)

        return scene_obs

    def load_mask_wrist(self, episode_id, step_id):
        """
        Load wrist camera mask from mask.npz file.
        
        Args:
            episode_id: Episode identifier
            step_id: Step identifier (should be string with zero padding)
            
        Returns:
            np.ndarray: Binary mask with shape (180, 320), dtype uint8
        """
        mask_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/mask.npz'
        if self.data_in_ceph:
            mask_bytes = self.client.get(mask_path, enable_cache=True)
            mask_data = np.load(io.BytesIO(mask_bytes))
        else:
            mask_data = np.load(mask_path)
        
        mask = mask_data['mask']  # Shape: (180, 320)
        return mask.astype(np.uint8)

    def load_mask_primary(self, episode_id, step_id):
        """
        Load primary camera mask. Since mask.npz only contains wrist mask,
        we return all zeros for primary camera.
        
        Args:
            episode_id: Episode identifier  
            step_id: Step identifier
            
        Returns:
            np.ndarray: All-zero mask with same shape as primary image
        """
        # Get primary image shape to match mask dimensions
        primary_image = self.load_primary_rgb(episode_id, step_id, self.primary_mode)
        height, width = primary_image.shape[:2]  # (H, W, C) -> (H, W)
        
        # Return all-zero mask with same spatial dimensions as primary image
        mask = np.zeros((height, width), dtype=np.uint8)
        return mask

    def load_contact_primary(self, episode_id, step_id):
        """
        Load primary camera contact point from waypoint.json file.
        
        Args:
            episode_id: Episode identifier
            step_id: Step identifier (should be string with zero padding)
            
        Returns:
            np.ndarray: Contact point with shape (2,), dtype float32
                       Contains [x, y] normalized coordinates in range [0, 1]
        """
        waypoint_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/waypoint.json'
        try:
            if self.data_in_ceph:
                waypoint_bytes = self.client.get(waypoint_path, enable_cache=True)
                waypoint_data = json.loads(waypoint_bytes.decode('utf-8'))
            else:
                with open(waypoint_path, 'r') as f:
                    waypoint_data = json.load(f)
            
            # Extract normalized contact point for primary camera
            contact_point = waypoint_data['image_primary']['contact_point']['normalized']
            return np.array(contact_point, dtype=np.float32)
            
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            # If waypoint file doesn't exist or is malformed, return default contact point
            logger.warning(f"Could not load contact point from {waypoint_path}: {e}")
            return np.array([0.5, 0.5], dtype=np.float32)  # Default to center

    def load_contact_wrist(self, episode_id, step_id):
        """
        Load wrist camera contact point from waypoint.json file.
        
        Args:
            episode_id: Episode identifier
            step_id: Step identifier (should be string with zero padding)
            
        Returns:
            np.ndarray: Contact point with shape (2,), dtype float32
                       Contains [x, y] normalized coordinates in range [0, 1]
        """
        waypoint_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/waypoint.json'
        try:
            if self.data_in_ceph:
                waypoint_bytes = self.client.get(waypoint_path, enable_cache=True)
                waypoint_data = json.loads(waypoint_bytes.decode('utf-8'))
            else:
                with open(waypoint_path, 'r') as f:
                    waypoint_data = json.load(f)
            
            # Extract normalized contact point for wrist camera
            contact_point = waypoint_data['image_wrist']['contact_point']['normalized']
            return np.array(contact_point, dtype=np.float32)
            
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            # If waypoint file doesn't exist or is malformed, return default contact point
            logger.warning(f"Could not load contact point from {waypoint_path}: {e}")
            return np.array([0.5, 0.5], dtype=np.float32)  # Default to center

    def load_spatial_primary(self, episode_id, step_id):
        """
        Load primary camera spatial point from spatialpoint.jsonl file.
        
        Args:
            episode_id: Episode identifier
            step_id: Step identifier (should be string with zero padding)
            
        Returns:
            np.ndarray: Spatial point with shape (2,), dtype float32
                       Contains [x, y] normalized coordinates in range [0, 1]
        """
        spatial_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/spatialpoint.jsonl'
        try:
            if self.data_in_ceph:
                spatial_bytes = self.client.get(spatial_path, enable_cache=True)
                spatial_lines = spatial_bytes.decode('utf-8').strip().split('\n')
            else:
                with open(spatial_path, 'r') as f:
                    spatial_lines = f.readlines()
            
            # Parse each line and find the primary camera data
            for line in spatial_lines:
                spatial_data = json.loads(line.strip())
                if spatial_data['image_type'] == 'image_primary':
                    spatial_point = spatial_data['final_point']
                    return np.array(spatial_point, dtype=np.float32)
            
            # If no primary camera data found, return default
            logger.warning(f"No primary camera spatial data found in {spatial_path}")
            return np.array([0.5, 0.5], dtype=np.float32)  # Default to center
            
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            # If spatial file doesn't exist or is malformed, return default spatial point
            logger.warning(f"Could not load spatial point from {spatial_path}: {e}")
            return np.array([0.5, 0.5], dtype=np.float32)  # Default to center

    def load_spatial_wrist(self, episode_id, step_id):
        """
        Load wrist camera spatial point from spatialpoint.jsonl file.
        
        Args:
            episode_id: Episode identifier
            step_id: Step identifier (should be string with zero padding)
            
        Returns:
            np.ndarray: Spatial point with shape (2,), dtype float32
                       Contains [x, y] normalized coordinates in range [0, 1]
        """
        spatial_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/spatialpoint.jsonl'
        try:
            if self.data_in_ceph:
                spatial_bytes = self.client.get(spatial_path, enable_cache=True)
                spatial_lines = spatial_bytes.decode('utf-8').strip().split('\n')
            else:
                with open(spatial_path, 'r') as f:
                    spatial_lines = f.readlines()
            
            # Parse each line and find the wrist camera data
            for line in spatial_lines:
                spatial_data = json.loads(line.strip())
                if spatial_data['image_type'] == 'image_wrist':
                    spatial_point = spatial_data['final_point']
                    return np.array(spatial_point, dtype=np.float32)
            
            # If no wrist camera data found, return default
            logger.warning(f"No wrist camera spatial data found in {spatial_path}")
            return np.array([0.5, 0.5], dtype=np.float32)  # Default to center
            
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            # If spatial file doesn't exist or is malformed, return default spatial point
            logger.warning(f"Could not load spatial point from {spatial_path}: {e}")
            return np.array([0.5, 0.5], dtype=np.float32)  # Default to center

    def load_traj_static(self, episode_id, step_id):
        """
        Load primary camera trajectory data from image_primary.npz file.
        
        Args:
            episode_id: Episode identifier
            step_id: Step identifier (should be string with zero padding)
            
        Returns:
            tuple: (tracks, visibility)
                - tracks: np.ndarray with shape (784, 2), dtype float32
                         Contains [Δx, Δy] MOTION DELTAS (NOT absolute coordinates!)
                         Normalized by image dimensions to range approximately [-1, 1]
                         IMPORTANT: Negative values indicate leftward/upward motion
                - visibility: np.ndarray with shape (784,), dtype float32
                            Visibility flag for each tracking point (0.0 or 1.0)
        """
        traj_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_primary.npz'
        try:
            if self.data_in_ceph:
                traj_bytes = self.client.get(traj_path, enable_cache=True)
                traj_data = np.load(io.BytesIO(traj_bytes))
            else:
                traj_data = np.load(traj_path)
            
            # Extract motion deltas (NOT absolute coordinates!)
            # tracks represent pixel displacement: [Δx, Δy] = [x_t+1 - x_t, y_t+1 - y_t]
            tracks = traj_data['tracks'].astype(np.float32)  # shape: (784, 2)
            visibility = traj_data['visibility'].astype(np.float32)  # shape: (784,)
            
            # Normalize motion deltas by image dimensions
            # This converts pixel displacements to relative motion in [-1, 1] range
            # Assuming image dimensions: primary camera is 320x180 (width x height)
            # Examples:
            #   - [10, 0]   -> [0.031, 0.0]    (rightward motion)
            #   - [-10, 0]  -> [-0.031, 0.0]   (leftward motion)
            #   - [0, -20]  -> [0.0, -0.111]   (upward motion)
            #   - [0, 20]   -> [0.0, 0.111]    (downward motion)
            tracks[:, 0] = tracks[:, 0] / 320.0  # Normalize Δx by image width
            tracks[:, 1] = tracks[:, 1] / 180.0  # Normalize Δy by image height
            
            # CRITICAL: Do NOT clip! Negative values represent leftward/upward motion
            # Clipping would destroy motion direction information
            
            return tracks, visibility
            
        except (FileNotFoundError, KeyError) as e:
            # If traj file doesn't exist or is malformed, return default values
            logger.warning(f"Could not load traj data from {traj_path}: {e}")
            # Return zero tracks (no motion) and zero visibility as default
            tracks = np.zeros((784, 2), dtype=np.float32)
            visibility = np.zeros(784, dtype=np.float32)
            return tracks, visibility

    def load_traj_gripper(self, episode_id, step_id):
        """
        Load wrist camera trajectory data from image_wrist.npz file.
        
        Args:
            episode_id: Episode identifier
            step_id: Step identifier (should be string with zero padding)
            
        Returns:
            tuple: (tracks, visibility)
                - tracks: np.ndarray with shape (784, 2), dtype float32
                         Contains [Δx, Δy] MOTION DELTAS (NOT absolute coordinates!)
                         Normalized by image dimensions to range approximately [-1, 1]
                         IMPORTANT: Negative values indicate leftward/upward motion
                - visibility: np.ndarray with shape (784,), dtype float32
                            Visibility flag for each tracking point (0.0 or 1.0)
        """
        traj_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.npz'
        try:
            if self.data_in_ceph:
                traj_bytes = self.client.get(traj_path, enable_cache=True)
                traj_data = np.load(io.BytesIO(traj_bytes))
            else:
                traj_data = np.load(traj_path)
            
            # Extract motion deltas (NOT absolute coordinates!)
            # tracks represent pixel displacement: [Δx, Δy] = [x_t+1 - x_t, y_t+1 - y_t]
            tracks = traj_data['tracks'].astype(np.float32)  # shape: (784, 2)
            visibility = traj_data['visibility'].astype(np.float32)  # shape: (784,)
            
            # Normalize motion deltas by image dimensions
            # This converts pixel displacements to relative motion in [-1, 1] range
            # Assuming image dimensions: wrist camera is 320x180 (width x height)
            # Examples:
            #   - [10, 0]   -> [0.031, 0.0]    (rightward motion)
            #   - [-10, 0]  -> [-0.031, 0.0]   (leftward motion)
            #   - [0, -20]  -> [0.0, -0.111]   (upward motion)
            #   - [0, 20]   -> [0.0, 0.111]    (downward motion)
            tracks[:, 0] = tracks[:, 0] / 320.0  # Normalize Δx by image width
            tracks[:, 1] = tracks[:, 1] / 180.0  # Normalize Δy by image height
            
            # CRITICAL: Do NOT clip! Negative values represent leftward/upward motion
            # Clipping would destroy motion direction information
            
            return tracks, visibility
            
        except (FileNotFoundError, KeyError) as e:
            # If traj file doesn't exist or is malformed, return default values
            logger.warning(f"Could not load traj data from {traj_path}: {e}")
            # Return zero tracks (no motion) and zero visibility as default
            tracks = np.zeros((784, 2), dtype=np.float32)
            visibility = np.zeros(784, dtype=np.float32)
            return tracks, visibility

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DistDroidDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseDroidDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]

    def __getitem__(self, idx):
        # 例如：idx = 65000

        # 1. 找到该索引属于哪个数据集
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        # bisect_right([50000, 80000, 100000], 65000) = 2
        # 说明样本65000属于第2个数据集（droid_failure）

        # 2. 计算在该数据集内的局部索引
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
            # local_idx = 65000 - 50000 = 15000
            # 说明是droid_failure数据集的第15000个样本
        else:
            local_idx = idx # 如果是第一个数据集，直接使用原索引

        # 3. 调用对应BaseDroidDataset的__getitem__方法
        return self.datasets[dataset_id].__getitem__(local_idx)
        # 等价于：self.datasets[2].__getitem__(15000)

    def collator(self, sample): # 批量处理batch
        # sample: 列表，包含batch_size个__getitem__的输出
        # 例如：如果batch_size = 4，sample就是4个上述格式字典的列表

        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        # 输入：4个 [16, 7] 的numpy数组
        # 输出：torch.Tensor [4, 16, 7] - 4个样本，每个16个时间步，每步7维动作

        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        # 输入：4个 [16, 8] 的numpy数组
        # 输出：torch.Tensor [4, 16, 8] - 4个样本，每个16个时间步，每步8维状态

        # 处理image_primary
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        # self.image_fn 是 preprocess_image_fn，使用ViT的image_processor
        # 对每个样本的16张PIL图像进行预处理：
        # - 调整大小到224x224
        # - 归一化到[-1, 1]
        # - 转换为tensor
        # 输入：4个样本，每个包含16张PIL图像
        # 输出：torch.Tensor [4, 16, 3, 224, 224]

        # 处理image_wrist
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        # 输入：4个样本，每个包含16张PIL图像
        # 输出：torch.Tensor [4, 16, 3, 224, 224] (或其他尺寸)


        stacked_language = [s["lang"] for s in sample] # 输入：["pick up the red cup", "open the drawer", "close the door", "move the bottle"]
        droid_episode_id = [s["droid_episode_id"] for s in sample]

        text_tensors = self.text_fn(stacked_language)
        # self.text_fn 是 preprocess_text_fn，使用CLIP的tokenizer
        # 输出：torch.Tensor [4, 77] - 4个样本，每个77个token（CLIP的固定长度）

        # 处理mask数据
        mask_static_tensors = torch.stack([s["mask_obs"]["mask_static"] for s in sample])
        # 输入：4个样本，每个包含 [T, H, W] 的mask tensor
        # 输出：torch.Tensor [4, T, H, W] - 4个样本，每个T个时间步，每步H×W的mask
        
        mask_gripper_tensors = torch.stack([s["mask_obs"]["mask_gripper"] for s in sample])
        # 输入：4个样本，每个包含 [T, H, W] 的mask tensor  
        # 输出：torch.Tensor [4, T, H, W] - 4个样本，每个T个时间步，每步H×W的mask

        # 处理contact point数据
        contact_primary_tensors = torch.stack([s["contact_obs"]["contact_primary"] for s in sample])
        # 输入：4个样本，每个包含 [T, 2] 的contact tensor
        # 输出：torch.Tensor [4, T, 2] - 4个样本，每个T个时间步，每步2维坐标(x, y)
        
        contact_wrist_tensors = torch.stack([s["contact_obs"]["contact_wrist"] for s in sample])
        # 输入：4个样本，每个包含 [T, 2] 的contact tensor
        # 输出：torch.Tensor [4, T, 2] - 4个样本，每个T个时间步，每步2维坐标(x, y)

        # 处理spatial point数据
        spatial_primary_tensors = torch.stack([s["spatial_obs"]["spatial_primary"] for s in sample])
        # 输入：4个样本，每个包含 [T, 2] 的spatial tensor
        # 输出：torch.Tensor [4, T, 2] - 4个样本，每个T个时间步，每步2维坐标(x, y)
        
        spatial_wrist_tensors = torch.stack([s["spatial_obs"]["spatial_wrist"] for s in sample])
        # 输入：4个样本，每个包含 [T, 2] 的spatial tensor
        # 输出：torch.Tensor [4, T, 2] - 4个样本，每个T个时间步，每步2维坐标(x, y)

        # 处理trajectory tracking数据
        traj_static_tensors = torch.stack([s["traj_obs"]["traj_static"] for s in sample])
        # 输入：4个样本，每个包含 [T, 784, 2] 的traj tensor
        # 输出：torch.Tensor [4, T, 784, 2] - 4个样本，每个T个时间步，每步784个跟踪点的(x, y)坐标
        
        visibility_static_tensors = torch.stack([s["traj_obs"]["visibility_static"] for s in sample])
        # 输入：4个样本，每个包含 [T, 784] 的visibility tensor
        # 输出：torch.Tensor [4, T, 784] - 4个样本，每个T个时间步，每步784个点的可见性标志
        
        traj_gripper_tensors = torch.stack([s["traj_obs"]["traj_gripper"] for s in sample])
        # 输入：4个样本，每个包含 [T, 784, 2] 的traj tensor
        # 输出：torch.Tensor [4, T, 784, 2] - 4个样本，每个T个时间步，每步784个跟踪点的(x, y)坐标
        
        visibility_gripper_tensors = torch.stack([s["traj_obs"]["visibility_gripper"] for s in sample])
        # 输入：4个样本，每个包含 [T, 784] 的visibility tensor
        # 输出：torch.Tensor [4, T, 784] - 4个样本，每个T个时间步，每步784个点的可见性标志

        # image_primary 图像增强处理
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2] # image_tensors.shape = [4, 16, 3, 224, 224] -> 4个样本，每个16个时间步，每张图像3通道224x224
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])

        # image_wrist 图像增强处理
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        robot_obs = torch.zeros(1)

        # 多步动作预测处理（如果启用）
        if self.act_step != 1: # 如果启用多步动作预测，例如act_step=3
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            # 新形状：[4, 16, 3, 7]
            # 4个样本，16个时间步，每步预测未来3步动作，每个动作7维

            # 步骤2：填充多步动作数据
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            # 样本0的动作序列（只显示前6个时间步）
            # action_tensors[0] = [
            #     [a0],  # 时间步0的动作
            #     [a1],  # 时间步1的动作
            #     [a2],  # 时间步2的动作
            #     [a3],  # 时间步3的动作
            #     [a4],  # 时间步4的动作
            #     [a5],  # 时间步5的动作
            #     # ... 共16个时间步
            # ]

            # 重组后的多步动作：
            # 样本0的多步动作（只显示前4个时间步）
            # actions[0] = [
            #     [[a0], [a1], [a2]],  # 时间步0：预测未来3步动作 a0,a1,a2
            #     [[a1], [a2], [a3]],  # 时间步1：预测未来3步动作 a1,a2,a3
            #     [[a2], [a3], [a4]],  # 时间步2：预测未来3步动作 a2,a3,a4
            #     [[a3], [a4], [a5]],  # 时间步3：预测未来3步动作 a3,a4,a5
            #     # ...
            # ]

            # 步骤3：创建多步状态张量
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            # 新形状：[4, 16, 3, 8]

            # 步骤4：填充多步状态数据
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]

            # 步骤5：状态维度处理
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            # 从8维状态 [x,y,z,rx,ry,rz,gripper_pos,gripper_state]
            # 提取为7维 [x,y,z,rx,ry,rz,gripper_state]
            # 保留前6维（6DOF位姿）和最后1维（夹爪状态），丢弃夹爪位置

            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
            # 对mask数据也进行相同的时序截断处理
            mask_static_tensors = mask_static_tensors[:, :-(self.act_step-1)]
            mask_gripper_tensors = mask_gripper_tensors[:, :-(self.act_step-1)]
            # 对contact数据也进行相同的时序截断处理
            contact_primary_tensors = contact_primary_tensors[:, :-(self.act_step-1)]
            contact_wrist_tensors = contact_wrist_tensors[:, :-(self.act_step-1)]
            # 对spatial数据也进行相同的时序截断处理
            spatial_primary_tensors = spatial_primary_tensors[:, :-(self.act_step-1)]
            spatial_wrist_tensors = spatial_wrist_tensors[:, :-(self.act_step-1)]
            # 对trajectory数据也进行相同的时序截断处理
            traj_static_tensors = traj_static_tensors[:, :-(self.act_step-1)]
            visibility_static_tensors = visibility_static_tensors[:, :-(self.act_step-1)]
            traj_gripper_tensors = traj_gripper_tensors[:, :-(self.act_step-1)]
            visibility_gripper_tensors = visibility_gripper_tensors[:, :-(self.act_step-1)]

        return (
            image_tensors,  # [4, 14, 3, 224, 224] - image_primary 图像
            text_tensors,  # [4, 77] - 文本token（不变）
            action_tensors,  # [4, 14, 3, 7] - 多步动作预测
            gripper_tensors,  # [4, 14, 3, 224, 224] - image_wrist 图像
            state_tensors,  # [4, 14, 8] - 状态序列
            robot_obs,  # [4, 14, 3, 7] - 多步状态序列
            mask_static_tensors,  # [4, 14, H, W] - primary camera mask
            mask_gripper_tensors,  # [4, 14, H, W] - wrist camera mask
            contact_primary_tensors,  # [4, 14, 2] - primary camera contact points
            contact_wrist_tensors,  # [4, 14, 2] - wrist camera contact points
            spatial_primary_tensors,  # [4, 14, 2] - primary camera spatial points
            spatial_wrist_tensors,  # [4, 14, 2] - wrist camera spatial points
            traj_static_tensors,  # [4, 14, 784, 2] - primary camera trajectory tracking
            visibility_static_tensors,  # [4, 14, 784] - primary camera visibility flags
            traj_gripper_tensors,  # [4, 14, 784, 2] - wrist camera trajectory tracking
            visibility_gripper_tensors  # [4, 14, 784] - wrist camera visibility flags
        )

def get_droid_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    # get_droid_dataset 主要用来加载各种参数, 包括text_fn(CLIP的tokenizer), image_fn(MAE ViT) 这些预处理器, 接入到DistDroidDataset
    dataset_names = [getattr(args, 'real_dataset_names', 'droid_success')]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    droid_dataset = DistDroidDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        #dataset_info=args.dataset_info,
        dataset_info=dataset_names[0],
        data_in_ceph=args.data_in_ceph,
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(droid_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        droid_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        droid_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=droid_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=droid_dataset)

class BaseLiberoDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config_libero,  # Use LIBERO-specific config without mask/contact/spatial/traj
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        dataset_info: str = "libero",
        small_size: int = 0,
        gripper_width: bool = False,
        load_libero_file: str = "h5", 
        **kwargs: Any,
    ):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        
        # Modified: Support multiple datasets with different JSON files
        # Original code (single dataset_info for all datasets):
        # assert os.path.exists(f"./data_info/{self.dataset_info}.json")
        # with open(f"./data_info/{self.dataset_info}.json", 'r') as f:
        
        # New code: Prioritize dataset_name.json, fallback to dataset_info.json
        json_path = f"./data_info/{self.dataset_name}.json"
        if not os.path.exists(json_path):
            json_path = f"./data_info/{self.dataset_info}.json"
        
        assert os.path.exists(json_path), f"JSON file not found: {json_path}"
        with open(json_path, 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]
        self.gripper_width = gripper_width
        self.load_libero_file = load_libero_file

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)

        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """

        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )

        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        episode_id = self.episode_list[episode_id]
        episodes = []
        language_instruction = None  # Initialize to avoid undefined variable
        for step_id in range(start_id, end_id):
            data_dict = {}
            str_step_id = str(step_id).zfill(4)
            if self.load_libero_file == "h5":
                other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/other.h5"
                with h5py.File(other_path, 'r') as other_file:
                    data_dict["rgb_static"] = self.load_primary_rgb(episode_id, str_step_id, self.primary_mode)
                    data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, str_step_id)
                    data_dict["rel_actions"] = self.load_action(other_file)
                    data_dict["robot_obs"] = self.load_robot_obs(other_file)
                    data_dict["scene_obs"] = self.load_scene_obs(episode_id, str_step_id)
                    # Load language instruction only once (use first step's instruction for entire sequence)
                    if language_instruction is None:
                        language_instruction = self.load_language_instruction(other_file, self.language_mode)
            elif self.load_libero_file == "npz":
                other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/other.npz"
                other_file = np.load(other_path, allow_pickle=True)
                data_dict["rgb_static"] = self.load_primary_rgb(episode_id, str_step_id, self.primary_mode)
                data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, str_step_id)
                data_dict["rel_actions"] = self.load_action(other_file)
                data_dict["robot_obs"] = self.load_robot_obs(other_file)
                data_dict["scene_obs"] = self.load_scene_obs(episode_id, str_step_id)
                # Load language instruction only once (use first step's instruction for entire sequence)
                if language_instruction is None:
                    language_instruction = self.load_language_instruction(other_file, self.language_mode)
            episodes.append(data_dict)
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = language_instruction  # Use the saved language instruction
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["episode_id"] = episode_id

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        image_primary = np.array(Image.open(image_primary_path).convert("RGB"))

        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg'
        image_wrist = np.array(Image.open(image_wrist_path).convert("RGB"))

        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_file, language_mode="language_instruction"):
        if self.load_libero_file == "h5":
            language_instruction = other_file[language_mode][()].decode('utf-8')
        elif self.load_libero_file == "npz":
            language_instruction = other_file[language_mode].tobytes().decode('utf-8')
        else:
            raise NotImplementedError

        return language_instruction
        
    def load_action(self, other_file, max_rel_pos=0.02, max_rel_orn=0.05, magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        if self.load_libero_file == "h5":
            action = other_file["action"][()]
        elif self.load_libero_file == "npz":
            action = other_file["action"]
        else:
            raise NotImplementedError
        
        return action

    def load_robot_obs(self, other_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        if self.load_libero_file == "h5":
            robot_obs[:6] = other_file['observation']['tcp_pose'][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file['observation']['gripper_state'][()]
            robot_obs[7:14] = other_file['observation']['proprio'][()]
            if self.gripper_width:
                robot_obs[-2:] = other_file['observation']['gripper_position'][()]
        elif self.load_libero_file == "npz":
            robot_obs[:6] = other_file["observation_tcp_pose"][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file["observation_gripper_state"]
            robot_obs[7:14] = other_file["observation_proprio"]
            if self.gripper_width:
                robot_obs[-2:] = other_file["observation_gripper_position"]
        else:
            raise NotImplementedError      

        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)

        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DiskLiberoDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        gripper_width: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseLiberoDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                gripper_width=gripper_width,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]
    
    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        episode_id = [s["episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)

        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)

        if self.act_step != 1:
        
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)

            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]

        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

def get_libero_pretrain_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = ["libero_90_converted"]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    libero_dataset = DiskLiberoDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info='libero_90_converted',
        gripper_width=args.gripper_width,
        load_libero_file=args.load_libero_file,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(libero_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        libero_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        libero_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=libero_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=libero_dataset)

def get_libero_finetune_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    # Modified: Use mixed datasets for targeted task improvement
    # Original single dataset options:
    # dataset_names = ["libero_10_converted"]  # Original full dataset
    # dataset_names = ["libero_10_rollouts_V3"]  # Previous collected data
    
    # New: Use args.libero_path to specify which dataset to use (for spatial, object, goal, etc.)
    # If libero_path is not provided, fall back to the original mixed dataset approach
    if hasattr(args, 'libero_path') and args.libero_path:
        dataset_names = [args.libero_path]
    else:
        # Fallback: Mixed training - original data + difficult tasks extra data
        dataset_names = ["libero_10_converted", "libero_10_part_converted"]
    
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    libero_dataset = DiskLiberoDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        # dataset_info is now auto-matched to dataset_name in BaseLiberoDataset
        # Use args.libero_path if available, otherwise fallback to libero_10_converted
        dataset_info=args.libero_path if (hasattr(args, 'libero_path') and args.libero_path) else 'libero_10_converted',
        gripper_width=args.gripper_width,
        load_libero_file=args.load_libero_file,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(libero_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        libero_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        libero_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=libero_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=libero_dataset)

class BaseRealDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        dataset_info: str = "",
        small_size: int = 0,
        gripper_width: bool = False,
        load_real_file: str = "npz", 
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        magic_scaling_factor_pos: float = 1.0,
        magic_scaling_factor_orn: float = 1.0,
        use_aug_data: bool = False,
        **kwargs: Any,
    ):
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/datasets/{dataset_name}'
        self.conf_path = '~/petreloss.conf'
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        self.use_aug_data = use_aug_data
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.magic_scaling_factor_pos = magic_scaling_factor_pos
        self.magic_scaling_factor_orn = magic_scaling_factor_orn
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        assert os.path.exists(f"./data_info/{self.dataset_info}.json")
        with open(f"./data_info/{self.dataset_info}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)
        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]
        self.gripper_width = gripper_width
        self.load_real_file = load_real_file

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        
        # Pad affordance labels if they exist (for BaseRealDataset)
        affordance_keys = [
            "mask_static", "mask_gripper",
            "contact_primary", "contact_wrist",
            "spatial_primary", "spatial_wrist",
            "traj_static", "traj_gripper",
            "visibility_static", "visibility_gripper"
        ]
        for aff_key in affordance_keys:
            if aff_key in seq:
                seq.update({aff_key: self._pad_with_repetition(seq[aff_key], pad_size, head)})

        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx
        head = False
        sequence = self._get_sequences(idx, window_size, head=head) # TODO

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)

        if self.use_aug_data:
            demo_list = self.episode_info_list[episode_id][2:]
            start_id, end_id = demo_list[start_id]
        episode_id = self.episode_list[episode_id] 
        exp_id = episode_id.split("/")[0]
        episodes = []
        for step_id in range(start_id, end_id):
            data_dict = {}
            try:
                str_step_id = str(step_id).zfill(4)
                if self.load_real_file == "npz":
                    other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/other.npz"
                    other_file = np.load(other_path, allow_pickle=True)
                else:  # h5
                    other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/other.h5"
                    other_file = h5py.File(other_path, 'r')
            except:
                print("episode_id :", episode_id)
                print("step_id :", str_step_id)
                print("num_step_per_episode :", num_step_per_episode)
                print("other_path", other_path)
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id, str_step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, str_step_id)
            data_dict["rel_actions"] = self.load_action(other_file, exp_id=exp_id)
            data_dict["robot_obs"] = self.load_robot_obs(other_file)
            data_dict["scene_obs"] = self.load_scene_obs(episode_id, str_step_id)
            
            # Load affordance labels from annotation files
            # Global Affordance: Load mask from mask.npz
            try:
                mask_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/mask.npz"
                mask_data = np.load(mask_path)
                # Resize mask to match expected dimensions (180x320 for masks)
                mask = mask_data['mask']  # Original shape from file
                data_dict["mask_static"] = mask.astype(np.uint8)
                data_dict["mask_gripper"] = mask.astype(np.uint8)  # Use same mask for wrist camera
            except Exception as e:
                # Use fallback: zero masks
                print(f"⚠️  Missing mask for episode {episode_id} step {str_step_id}: {e}")
                data_dict["mask_static"] = np.zeros((180, 320), dtype=np.uint8)
                data_dict["mask_gripper"] = np.zeros((180, 320), dtype=np.uint8)
            
            # Local Affordance: Load contact point from waypoint.json
            try:
                waypoint_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/waypoint.json"
                with open(waypoint_path, 'r') as f:
                    waypoint_data = json.load(f)
                # Extract normalized contact points [x, y] in [0, 1] range
                contact_primary_norm = waypoint_data['image_primary']['contact_point']['normalized']
                contact_wrist_norm = waypoint_data['image_wrist']['contact_point']['normalized']
                data_dict["contact_primary"] = np.array(contact_primary_norm, dtype=np.float32)  # Shape: (2,)
                data_dict["contact_wrist"] = np.array(contact_wrist_norm, dtype=np.float32)      # Shape: (2,)
            except Exception as e:
                # Use fallback: center point
                print(f"⚠️  Missing waypoint for episode {episode_id} step {str_step_id}: {e}")
                data_dict["contact_primary"] = np.array([0.5, 0.5], dtype=np.float32)
                data_dict["contact_wrist"] = np.array([0.5, 0.5], dtype=np.float32)
            
            # Spatial Affordance: Load spatial point from spatialpoint.jsonl
            try:
                spatial_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/spatialpoint.jsonl"
                with open(spatial_path, 'r') as f:
                    lines = f.readlines()
                # Parse primary and wrist spatial points
                spatial_primary = None
                spatial_wrist = None
                for line in lines:
                    spatial_data = json.loads(line)
                    if spatial_data['image_type'] == 'image_primary':
                        spatial_primary = spatial_data['final_point']  # [x, y]
                    elif spatial_data['image_type'] == 'image_wrist':
                        spatial_wrist = spatial_data['final_point']
                
                data_dict["spatial_primary"] = np.array(spatial_primary if spatial_primary else [0.5, 0.5], dtype=np.float32)  # Shape: (2,)
                data_dict["spatial_wrist"] = np.array(spatial_wrist if spatial_wrist else [0.5, 0.5], dtype=np.float32)        # Shape: (2,)
            except Exception as e:
                # Use fallback: center point
                print(f"⚠️  Missing spatialpoint for episode {episode_id} step {str_step_id}: {e}")
                data_dict["spatial_primary"] = np.array([0.5, 0.5], dtype=np.float32)
                data_dict["spatial_wrist"] = np.array([0.5, 0.5], dtype=np.float32)
            
            # Dynamic Affordance: Load trajectories from cotracker.json
            try:
                cotracker_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/cotracker.json"
                with open(cotracker_path, 'r') as f:
                    cotracker_data = json.load(f)
                # Extract tracks: 784 points x 2 coordinates (motion/displacement vectors)
                tracks_primary = np.array(cotracker_data['image_primary']['full_tracks'], dtype=np.float32)     # Shape: (784, 2)
                tracks_wrist = np.array(cotracker_data['image_wrist']['full_tracks'], dtype=np.float32)         # Shape: (784, 2)
                visibility_primary = np.array(cotracker_data['image_primary']['full_visibility'], dtype=np.float32)  # Shape: (784,)
                visibility_wrist = np.array(cotracker_data['image_wrist']['full_visibility'], dtype=np.float32)      # Shape: (784,)
                
                data_dict["traj_static"] = tracks_primary        # Shape: (784, 2)
                data_dict["traj_gripper"] = tracks_wrist         # Shape: (784, 2)
                data_dict["visibility_static"] = visibility_primary   # Shape: (784,)
                data_dict["visibility_gripper"] = visibility_wrist    # Shape: (784,)
            except Exception as e:
                # Use fallback: zero trajectories
                print(f"⚠️  Missing cotracker for episode {episode_id} step {str_step_id}: {e}")
                data_dict["traj_static"] = np.zeros((784, 2), dtype=np.float32)
                data_dict["traj_gripper"] = np.zeros((784, 2), dtype=np.float32)
                data_dict["visibility_static"] = np.zeros((784,), dtype=np.float32)
                data_dict["visibility_gripper"] = np.zeros((784,), dtype=np.float32)
            
            episodes.append(data_dict)
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = self.load_language_instruction(other_file, self.language_mode)
        
        # Stack affordance labels if they exist in episodes
        affordance_keys = [
            "mask_static", "mask_gripper",
            "contact_primary", "contact_wrist",
            "spatial_primary", "spatial_wrist",
            "traj_static", "traj_gripper",
            "visibility_static", "visibility_gripper"
        ]
        for aff_key in affordance_keys:
            if aff_key in episodes[0]:  # Check if first episode has this key
                try:
                    episode[aff_key] = np.stack([ep[aff_key] for ep in episodes])
                except Exception as e:
                    print(f"⚠️  Failed to stack {aff_key}: {e}")
        
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)

        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["episode_id"] = episode_id
        
        # Add stacked affordance labels to seq_dict
        for aff_key in affordance_keys:
            if aff_key in episode:
                seq_dict[aff_key] = torch.from_numpy(episode[aff_key])

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        image_primary = np.array(Image.open(image_primary_path).convert("RGB"))
        
        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg'
        image_wrist = np.array(Image.open(image_wrist_path).convert("RGB"))
        
        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_file, language_mode="language_instruction"):
        if self.load_real_file == "npz":
            language_instruction = other_file[language_mode].tobytes().decode('utf-8')
        else:  # h5
            lang_data = other_file[language_mode][()]
            if isinstance(lang_data, bytes):
                language_instruction = lang_data.decode('utf-8')
            else:
                language_instruction = str(lang_data)
        
        return language_instruction

    def load_action(self, other_file, exp_id):
        if self.load_real_file == "npz":
            action = other_file["delta_cur_2_last_action"]
            action[:3] /= (self.max_rel_pos * self.magic_scaling_factor_pos)
            action[3:6] /= (self.max_rel_orn * self.magic_scaling_factor_orn)
        else:  # h5
            action = other_file["action_delta_wrist_pose"][()]
            action[:3] /= (self.max_rel_pos * self.magic_scaling_factor_pos)
            action[3:6] /= (self.max_rel_orn * self.magic_scaling_factor_orn)
        
        return action

    def load_robot_obs(self, other_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        if self.load_real_file == "npz":
            robot_obs[:6] = other_file["gripper_pose"]
            robot_obs[-1] = other_file["gripper_open_state"]
            robot_obs[7:14] = other_file["joints"]
        else:  # h5
            robot_obs[:6] = other_file["observation/gripper_pose6d"][()]
            robot_obs[-1] = other_file["observation/gripper_open_state"][()][0] if len(other_file["observation/gripper_open_state"][()]) > 0 else other_file["observation/gripper_open_state"][()]
            robot_obs[7:14] = other_file["observation/joint_position"][()]
        
        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)
        
        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DiskRealDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        gripper_width: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
                BaseRealDataset(
                    *args, 
                    dataset_name=dataset_name,
                    act_step=act_step,
                    small_size=small_size,
                    gripper_width=gripper_width,
                    **kwargs,
                    
                ) for dataset_name in dataset_names
            ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]

    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)

    def collator(self, sample):
        # All samples now have complete affordance labels (with fallbacks if files are missing)
        # No need to filter anymore
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        episode_id = [s["episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        
        # Load affordance labels (guaranteed to exist for valid samples)
        # All samples have been padded to the same length, so we can stack them
        # Global Affordance: masks
        mask_tensors_primary = torch.stack([s["mask_static"] for s in sample])  # [B, T, H, W]
        mask_tensors_wrist = torch.stack([s["mask_gripper"] for s in sample])   # [B, T, H, W]
        
        # Local Affordance: contact points
        contact_tensors_primary = torch.stack([s["contact_primary"] for s in sample])  # [B, T, 2]
        contact_tensors_wrist = torch.stack([s["contact_wrist"] for s in sample])      # [B, T, 2]
        
        # Spatial Affordance: spatial points
        spatial_tensors_primary = torch.stack([s["spatial_primary"] for s in sample])  # [B, T, 2]
        spatial_tensors_wrist = torch.stack([s["spatial_wrist"] for s in sample])      # [B, T, 2]
        
        # Dynamic Affordance: trajectories
        traj_tensors_static = torch.stack([s["traj_static"] for s in sample])           # [B, T, 784, 2]
        visibility_static = torch.stack([s["visibility_static"] for s in sample])       # [B, T, 784]
        traj_tensors_gripper = torch.stack([s["traj_gripper"] for s in sample])         # [B, T, 784, 2]
        visibility_gripper = torch.stack([s["visibility_gripper"] for s in sample])     # [B, T, 784]

        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)

        if self.act_step != 1:
        
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
            
            # Also trim affordance labels if they exist
            if mask_tensors_primary is not None:
                mask_tensors_primary = mask_tensors_primary[:, :-(self.act_step-1)]
                mask_tensors_wrist = mask_tensors_wrist[:, :-(self.act_step-1)]
            if contact_tensors_primary is not None:
                contact_tensors_primary = contact_tensors_primary[:, :-(self.act_step-1)]
                contact_tensors_wrist = contact_tensors_wrist[:, :-(self.act_step-1)]
            if spatial_tensors_primary is not None:
                spatial_tensors_primary = spatial_tensors_primary[:, :-(self.act_step-1)]
                spatial_tensors_wrist = spatial_tensors_wrist[:, :-(self.act_step-1)]
            if traj_tensors_static is not None:
                traj_tensors_static = traj_tensors_static[:, :-(self.act_step-1)]
                visibility_static = visibility_static[:, :-(self.act_step-1)]
                traj_tensors_gripper = traj_tensors_gripper[:, :-(self.act_step-1)]
                visibility_gripper = visibility_gripper[:, :-(self.act_step-1)]

        # Return basic data + affordance labels (16 elements total)
        return (image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs,
                mask_tensors_primary, mask_tensors_wrist,
                contact_tensors_primary, contact_tensors_wrist,
                spatial_tensors_primary, spatial_tensors_wrist,
                traj_tensors_static, visibility_static, traj_tensors_gripper, visibility_gripper) 

def get_real_finetune_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = [args.real_dataset_names]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    real_dataset = DiskRealDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info=args.dataset_info,  # Use dataset_info instead of real_dataset_names
        gripper_width=args.gripper_width,
        load_real_file="h5",  # Changed from npz to h5
        use_aug_data=args.use_aug_data,
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(real_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        real_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=real_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=real_dataset)

def get_real_val_dataset(args, image_processor, tokenizer, epoch=0):
    """
    Get validation dataset for real-world data
    
    Args:
        args: Training arguments (需要有 real_val_dataset_names 或默认使用 'real_world_val')
        image_processor: Image preprocessor
        tokenizer: Text tokenizer
        epoch: Current epoch
    
    Returns:
        DataInfo object with validation dataloader
    """
    # 使用验证集配置文件名称（默认 real_world_val）
    val_dataset_info = getattr(args, 'real_val_dataset_names', 'real_world_val')
    # 使用与训练集相同的数据集目录名称
    dataset_names = [args.real_dataset_names]
    
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    
    real_val_dataset = DiskRealDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=-1,  # 验证时不使用数据增强
        gripper_pad=-1,  # 验证时不使用数据增强
        traj_cons=args.traj_cons,
        text_aug=False,  # 验证时不使用文本增强
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=0,  # 验证时使用全部数据
        dataset_info=val_dataset_info,
        gripper_width=args.gripper_width,
        load_real_file="h5",  # Changed from npz to h5
        use_aug_data=False,  # 验证时不使用增强数据
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    
    num_samples = len(real_val_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_workers = max(1, args.workers)
    
    # 验证集使用DistributedSampler但不shuffle
    sampler = DistributedSampler(
        real_val_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,  # 验证时不打乱
        seed=args.seed,
        drop_last=False,  # 验证时不丢弃最后一批
    )
    
    dataloader = DataLoader(
        real_val_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=False,  # 验证时不需要持久化workers
        collate_fn=real_val_dataset.collator,
        drop_last=False  # 验证时不丢弃最后一批
    )
    
    dataloader.num_batches = len(dataloader)
    dataloader.num_samples = num_samples
    
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=real_val_dataset)

class BaseOXEDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        small_size: int = 0,
        data_in_ceph: bool = False, 
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        magic_scaling_factor_pos: float = 1.0,
        magic_scaling_factor_orn: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        
        self.dataset_name = dataset_name 
        self.dataset_info = dataset_name
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        self.data_in_ceph = data_in_ceph
        if self.data_in_ceph:
            self.client = Client(self.conf_path)
        else:
            self.client = None
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        self.data_in_ceph = data_in_ceph
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.magic_scaling_factor_pos = magic_scaling_factor_pos
        self.magic_scaling_factor_orn = magic_scaling_factor_orn
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        if self.data_in_ceph:
            assert self.client.isdir(self.dataset_path)
        else:
            os.path.exists(self.dataset_path)
        assert os.path.exists(f"data_info/{self.dataset_info}.json")
        with open(f"data_info/{self.dataset_info}.json", 'r') as f: #TODO
            self.episode_info_list = json.load(f)
            # Check if this is OXE format (first element is a dict) or simple format (all elements are lists)
            if self.episode_info_list and isinstance(self.episode_info_list[0], dict):
                # OXE format: [{"dataset_name": "...", "wrist_image": "..."}, ["episodes/000000", 71], ...]
                self.episode_list = [f[0] for f in self.episode_info_list[1:]]
                self.transform_wrist_image = self.episode_info_list[0].get("wrist_image", None)
                self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list[1:]]
            else:
                # Simple format: [["000000", 982], ["000001", 956], ...]
                self.episode_list = [f[0] for f in self.episode_info_list]
                self.transform_wrist_image = None
                self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)
        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        
        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        
        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list
        
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        episode_id = self.episode_list[episode_id]
        episodes = []
        for step_id in range(start_id, end_id):
            data_dict = {}
            try:
                step_id = str(step_id).zfill(4)
                other_path = f"{self.dataset_path}/{episode_id}/steps/{step_id}/other.h5"
                if self.data_in_ceph:
                    other_bytes = self.client.get(other_path, enable_cache=True)
                    other_h5_file = h5py.File(io.BytesIO(other_bytes))
                else:
                    other_h5_file = h5py.File(other_path)
            except:
                print("episode_id :", episode_id)
                print("step_id :", step_id)
                print("num_step_per_episode :", num_step_per_episode)
                print("other_path", f"{self.dataset_path}/{episode_id}/steps/{step_id}/other.h5")
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id, step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, step_id)
            data_dict["rel_actions"] = self.load_action(other_h5_file)
            try:
                data_dict["robot_obs"] = self.load_robot_obs(other_h5_file)
            except:
                print(f"{self.dataset_path}/{episode_id}/steps/{step_id}/other.h5")
                raise NotImplementedError
            data_dict["scene_obs"] = self.load_scene_obs(episode_id, step_id)
            episodes.append(data_dict)
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = self.load_language_instruction(other_h5_file, self.language_mode)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["droid_episode_id"] = episode_id

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary = f'{self.dataset_path}/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        if self.data_in_ceph:
            image_primary = self.client.get(image_primary, enable_cache=True)
            image_primary = io.BytesIO(image_primary)
        image_primary = np.array(Image.open(image_primary).convert("RGB")) 
        
        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist = f'{self.dataset_path}/{episode_id}/steps/{step_id}/image_wrist.jpg'
        if self.data_in_ceph:
            image_wrist = self.client.get(image_wrist, enable_cache=True)
            image_wrist = io.BytesIO(image_wrist)
        image_wrist = np.array(Image.open(image_wrist).convert("RGB")) 
        if self.transform_wrist_image ==  "Flip vertically & horizontally":
            image_wrist = np.flip(image_wrist, axis=1) 
            image_wrist = np.flip(image_wrist, axis=0)
        
        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_h5_file, language_mode="language_instruction"):
        language_instruction = other_h5_file[language_mode][()].decode('utf-8')
        
        return language_instruction
        
    def load_action(self, other_h5_file, max_rel_pos=0.02, max_rel_orn=0.05, magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        action = other_h5_file["action_delta_wrist_pose"][()]
        if self.dataset_name in [
            f"furniture_bench_dataset_converted_externally_to_rlds",
            f"berkeley_autolab_ur5",
            f"berkeley_fanuc_manipulation",
            ]:
            action[:3] /= (self.max_rel_pos * 10.0)
            action[3:6] /= (self.max_rel_orn * 10.0)
        else:
            action[:3] /= (self.max_rel_pos * self.magic_scaling_factor_pos)
            action[3:6] /= (self.max_rel_orn * self.magic_scaling_factor_orn)

        return action

    def load_robot_obs(self, other_h5_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        robot_obs[:6] = other_h5_file['observation']['gripper_pose6d'][()]
        robot_obs[-1] = other_h5_file['observation']['gripper_open_state'][()][0]
        if self.dataset_name in ["berkeley_autolab_ur5", "berkeley_fanuc_manipulation", "jaco_play"]:
            pass 
        else:
            robot_obs[7:14] = other_h5_file['observation']['joint_position'][()]
        
        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)
        
        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DistOXEDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseOXEDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]
    
    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)


    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        droid_episode_id = [s["droid_episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        robot_obs = torch.zeros(1)
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]

        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

def get_oxe_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    # Allow custom dataset names from args, fallback to default OXE datasets
    if hasattr(args, 'real_dataset_names') and args.real_dataset_names:
        dataset_names = [args.real_dataset_names]
    else:
        dataset_names = [
            ## other robots
            "berkeley_autolab_ur5",
            "jaco_play",
            ## franka
            "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
            "viola",
            "stanford_hydra_dataset_converted_externally_to_rlds",
            "berkeley_fanuc_manipulation",
            "austin_buds_dataset_converted_externally_to_rlds",
            "utaustin_mutex",
            "taco_play",
            "austin_sailor_dataset_converted_externally_to_rlds",
            "austin_sirius_dataset_converted_externally_to_rlds",
            "furniture_bench_dataset_converted_externally_to_rlds",
        ]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    oxe_dataset = DistOXEDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        data_in_ceph=args.data_in_ceph,
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(oxe_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        oxe_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        oxe_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=oxe_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=oxe_dataset)


if __name__ == "__main__":
    from arguments_utils import get_args_and_cfg
    from tqdm import tqdm
    args, _ = get_args_and_cfg()
    device='cuda'
    model, image_processor = clip.load("ViT-B/32", device=device)
    calvin_dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0)
    calvin_dataset.set_epoch(epoch=0)
    calvin_loader = calvin_dataset.dataloader
    num_batches_per_epoch = calvin_loader.num_batches
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
    )
    mv_avg_loss = []
    for num_steps, batch in t:
        if num_steps > 0:
            torch.cuda.synchronize()
            t2 = time.time()
            print("t2 - t1", t2 - t1)
        torch.cuda.synchronize()
        t1 = time.time()
