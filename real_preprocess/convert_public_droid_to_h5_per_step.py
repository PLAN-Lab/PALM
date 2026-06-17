import os

import random
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import tensorflow as tf
import dlimp as dl
import tensorflow_datasets as tfds
import h5py
import tqdm
import argparse
from pathlib import Path
import yaml
from PIL import Image
import pybullet as p
import time

from pdb import set_trace
import math

### Rotation ###
from scipy.spatial.transform import Rotation

### mujoco ###
import mujoco
import mujoco.viewer
from pdb import set_trace
from copy import deepcopy

tf.config.set_visible_devices([], "GPU")


def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    return np.array([[np.cos(q), -np.sin(q), 0, a],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])


def get_fk_solution(joint_angles):
    dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi / 2, joint_angles[1]],
                 [0, 0.316, np.pi / 2, joint_angles[2]],
                 [0.0825, 0, np.pi / 2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                 [0, 0, np.pi / 2, joint_angles[5]],
                 [0.088, 0, np.pi / 2, joint_angles[6]],
                 [0, 0.107, 0, 0],
                 [0, 0, 0, -np.pi / 4],
                 [0.0, 0.1034, 0, 0]]

    T = np.eye(4)
    for i in range(8):
        T = T @ get_tf_mat(i, dh_params)
    return T


def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = Rotation.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose


def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


def convert_action(actions, gripper_positions, start_gripper_pose):
    ### action_tcp_poses, action_wrist_poses, action_delta_tcp_poses, action_delta_wrist_poses = convert_action(actions, gripper_positions） ###
    # actions.shape: [episode_length, 6]
    # gripper_positions.shape: [episode_length, 1]
    # start_gripper_pose.shape : [4, 4]

    # action_tcp_poses.shape: [episode_length, 6]
    # action_wrist_poses.shape: [episode_length, 6]
    # action_delta_tcp_poses.shape: [episode_length, 6]
    # action_delta_wrist_poses.shape: [episode_length, 6]

    peijian_h = 0.0117

    episode_length, _ = actions.shape
    action_tcp_poses = np.zeros((episode_length, 7))
    action_wrist_poses = np.zeros((episode_length, 7))
    action_delta_tcp_poses = np.zeros((episode_length, 7))
    action_delta_wrist_poses = np.zeros((episode_length, 7))

    ### mujoco setup ###
    m = mujoco.MjModel.from_xml_path(f"mujoco_menagerie/robotiq_2f85/2f85.xml")
    d = mujoco.MjData(m)

    ###
    ### gripper command ###
    gripper_commands = []
    for i in range(gripper_positions.shape[0]):
        if i == 0:
            gripper_command = 1
        elif gripper_positions[i, -1] > gripper_positions[i - 1, -1]:
            gripper_command = -1
        elif gripper_positions[i, -1] < gripper_positions[i - 1, -1]:
            gripper_command = 1
        elif gripper_positions[i, -1] == gripper_positions[i - 1, -1]:
            gripper_command = gripper_command_old
        gripper_command_old = gripper_command
        gripper_commands.append(np.array([gripper_command]))
    gripper_commands = np.array(gripper_commands)
    action_tcp_poses[:, -1] = gripper_commands[:, 0]
    action_wrist_poses[:, -1] = gripper_commands[:, 0]
    action_delta_tcp_poses[:, -1] = gripper_commands[:, 0]
    action_delta_wrist_poses[:, -1] = gripper_commands[:, 0]

    ### wrist pos & rot ###
    action_wrist_poses[:, :6] = actions[:, :6].copy()
    action_wrist_poses[:, -1] = gripper_commands[:, 0]
    for j in range(action_delta_wrist_poses.shape[0]):
        if j == 0:
            last2world = start_gripper_pose
        else:
            last2world = _6d_to_pose(action_wrist_poses[j - 1, :6], degrees=False)
        cur2world = _6d_to_pose(action_wrist_poses[j, :6], degrees=False)
        cur2last = np.linalg.inv(last2world) @ cur2world
        action_delta_wrist_poses[j, :6] = pose_to_6d(cur2last)

    action_tcp_poses[:, 3:6] = actions[:, 3:6]

    ### tcp pos ###
    wrist_rot = Rotation.from_euler("xyz", actions[:, 3:6]).as_matrix()
    tcp_res = []
    for i in range(actions.shape[0]):
        mujoco.mj_resetData(m, d)
        d.ctrl = gripper_positions[i, 0] * 255
        for _ in range(250):
            mujoco.mj_step(m, d)
            mujoco.mj_kinematics(m, d)
        pad2_z = d.geom('right_pad2').xpos[2]
        tcp_res.append([0.0, 0.0, pad2_z])
    tcp_res = np.array(tcp_res)
    action_tcp_res = deepcopy(tcp_res)
    tcp_res[:, 2] += peijian_h
    tcp_pos = np.einsum('ijk,ik->ij', wrist_rot, tcp_res) + actions[:, :3]
    action_tcp_poses[:, :3] = tcp_pos

    ### delta tcp action ###
    mujoco.mj_resetData(m, d)
    d.ctrl = gripper_positions[0, 0] * 255
    for _ in range(250):
        mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)

    pad2_z = d.geom('right_pad2').xpos[2]
    start_tcp_res = np.array([0.0, 0.0, pad2_z])
    start_tcp_res[2] += peijian_h
    start_tcp_pos = start_gripper_pose[:3, :3] @ start_tcp_res + start_gripper_pose[:3, 3]
    start_tcp_pose = np.eye(4)
    start_tcp_pose[:3, :3] = start_gripper_pose[:3, :3]
    start_tcp_pose[:3, 3] = start_tcp_pos

    for j in range(action_delta_tcp_poses.shape[0]):
        if j == 0:
            last2world = start_tcp_pose
        else:
            last2world = _6d_to_pose(action_tcp_poses[j - 1, :6], degrees=False)
        cur2world = _6d_to_pose(action_tcp_poses[j, :6], degrees=False)
        cur2last = np.linalg.inv(last2world) @ cur2world
        action_delta_tcp_poses[j, :6] = pose_to_6d(cur2last)

    return action_tcp_poses, action_wrist_poses, action_delta_tcp_poses, action_delta_wrist_poses, action_tcp_res


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def to_np_dtype(maybe_tf_dtype):
    dtype_map = {
        tf.bool: np.dtype('bool'),
        tf.string: str,
        tf.float16: np.float16,
        tf.float32: np.float32,
        tf.float64: np.float64,
        tf.int8: np.int8,
        tf.int16: np.int16,
        tf.int32: np.int32,
        tf.int64: np.int64,
        tf.uint8: np.uint8,
        tf.uint16: np.uint16,
    }

    # keep the unfound the same
    np_dtype = dtype_map.get(maybe_tf_dtype, maybe_tf_dtype)

    return np_dtype


class DatasetConverter:
    def __init__(
            self,
            src_dir: str,
            tgt_dir: str,
            rank: int,
            num_worker: int,
            start_episode_idx,
            end_episode_idx,
            flag,
    ):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.rank = rank
        self.num_worker = num_worker
        self.start_episode_idx = start_episode_idx
        self.end_episode_idx = end_episode_idx
        self.flag = flag
        self.gripper_open_criterion = 13 / 255

    def process_episode(self, episode_dir, episode, episode_index, flag):
        # get success or not
        path_name = episode['traj_metadata']['episode_metadata']['file_path'][0].numpy().decode('utf-8')
        if flag in path_name:
            pass
        else:
            return

        # get episode length
        num_steps = episode['action'].shape[0]

        # get language instruction
        language_instructions = episode['language_instruction']
        language_instructions_2 = episode['language_instruction_2']
        language_instructions_3 = episode['language_instruction_3']

        debug_lang1 = np.array(language_instructions[0], dtype=h5py.string_dtype(encoding='utf-8'))
        debug_lang2 = np.array(language_instructions_2[0], dtype=h5py.string_dtype(encoding='utf-8'))
        debug_lang3 = np.array(language_instructions_3[0], dtype=h5py.string_dtype(encoding='utf-8'))

        episode_dir = episode_dir / str(episode_index).zfill(6)
        episode_dir.mkdir(exist_ok=True)

        # save episode length and language instruction
        with h5py.File(f'{episode_dir}/meta_info.h5', 'w') as h5_file:
            h5_file.create_dataset(name='length', data=num_steps)

        ## joint positions ##
        joint_positions = episode['observation']['joint_position']
        joint_positions = np.array(joint_positions).astype(to_np_dtype(joint_positions.dtype))

        ## gripper positions ##
        gripper_positions = episode['observation']['gripper_position']
        gripper_positions = np.array(gripper_positions).astype(to_np_dtype(gripper_positions.dtype))
        gripper_open_state = np.ones_like(gripper_positions) * (-1.0)
        gripper_open_state[gripper_positions < self.gripper_open_criterion] = 1.0

        ## gripper poses ##
        gripper_pose = np.concatenate(
            [get_fk_solution(joint_positions[i])[None, ...] for i in range(joint_positions.shape[0])], axis=0)
        gripper_pose6d = np.concatenate(
            [pose_to_6d(gripper_pose[i])[None, ...] for i in range(joint_positions.shape[0])], axis=0)

        ## actions ##
        action_gripper_velocity = episode["action_dict"]["gripper_velocity"]
        action_gripper_velocity = np.array(action_gripper_velocity).astype(to_np_dtype(action_gripper_velocity.dtype))
        action_joint_position = episode["action_dict"]["joint_position"]
        action_joint_position = np.array(action_joint_position).astype(to_np_dtype(action_joint_position.dtype))
        action_joint_velocity = episode["action_dict"]["joint_velocity"]
        action_joint_velocity = np.array(action_joint_velocity).astype(to_np_dtype(action_joint_velocity.dtype))
        action_cartesian_velocity = episode["action_dict"]["cartesian_velocity"]
        action_cartesian_velocity = np.array(action_cartesian_velocity).astype(
            to_np_dtype(action_cartesian_velocity.dtype))
        action_cartesian_position = episode["action_dict"]["cartesian_position"]
        action_cartesian_position = np.array(action_cartesian_position).astype(
            to_np_dtype(action_cartesian_position.dtype))

        action_tcp_poses, action_wrist_poses, action_delta_tcp_poses, action_delta_wrist_poses, action_tcp_res = \
            convert_action(action_cartesian_position, gripper_positions, gripper_pose[0])

        steps_dir = episode_dir / 'steps'
        steps_dir.mkdir(exist_ok=True)
        for step_index in range(num_steps):
            step_dir = episode_dir / 'steps' / str(step_index).zfill(4)
            step_dir.mkdir(exist_ok=True)

            with h5py.File(f'{step_dir}/other.h5', 'w') as h5_file:
                # language instruction
                h5_file.create_dataset('language_instruction', data=np.array(language_instructions[step_index],
                                                                             dtype=h5py.string_dtype(encoding='utf-8')))
                h5_file.create_dataset('language_instruction_2', data=np.array(language_instructions_2[step_index],
                                                                               dtype=h5py.string_dtype(
                                                                                   encoding='utf-8')))
                h5_file.create_dataset('language_instruction_3', data=np.array(language_instructions_3[step_index],
                                                                               dtype=h5py.string_dtype(
                                                                                   encoding='utf-8')))

                # episode length
                h5_file.create_dataset(name='episode_length', data=num_steps)

                # timestep
                h5_file.create_dataset(name='timestep', data=step_index)

                # action
                h5_file.create_dataset(name='action_delta_tcp_pose', data=action_delta_tcp_poses[step_index])
                h5_file.create_dataset(name='action_delta_wrist_pose', data=action_delta_wrist_poses[step_index])
                h5_file.create_dataset(name='action_tcp_pose', data=action_tcp_poses[step_index])
                h5_file.create_dataset(name='action_wrist_pose', data=action_wrist_poses[step_index])
                h5_file.create_dataset(name='action_tcp_res', data=action_tcp_res[step_index])
                h5_file.create_dataset(name='gripper_command', data=action_delta_tcp_poses[step_index, -1])

                h5_file.create_dataset(name="action_gripper_velocity", data=action_gripper_velocity[step_index])
                h5_file.create_dataset(name="action_joint_position", data=action_joint_position[step_index])
                h5_file.create_dataset(name="action_joint_velocity", data=action_joint_velocity[step_index])
                h5_file.create_dataset(name="action_cartesian_velocity", data=action_cartesian_velocity[step_index])

                # observation (timestep, proprio, image_XXX)
                observation_group = h5_file.create_group(name='observation')

                ## image
                ### image_primary
                data = episode['observation']['exterior_image_1_left'][step_index]
                data = tf.io.decode_image(data, expand_animations=False, dtype=tf.uint8)
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                Image.fromarray(np_data).save(f'{step_dir}/image_primary.jpg')
                ### image_primary
                data = episode['observation']['wrist_image_left'][step_index]
                data = tf.io.decode_image(data, expand_animations=False, dtype=tf.uint8)
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                Image.fromarray(np_data).save(f'{step_dir}/image_wrist.jpg')
                ### image_primary
                data = episode['observation']['exterior_image_2_left'][step_index]
                data = tf.io.decode_image(data, expand_animations=False, dtype=tf.uint8)
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                Image.fromarray(np_data).save(f'{step_dir}/image_3.jpg')

                ## joint position
                observation_group.create_dataset(name='joint_position', data=joint_positions[step_index])

                ## gripper position
                observation_group.create_dataset(name='gripper_position', data=gripper_positions[step_index])

                ## gripper open state
                observation_group.create_dataset(name="gripper_open_state", data=gripper_open_state[step_index])

                ## gripper pose ###
                observation_group.create_dataset(name="gripper_pose6d", data=gripper_pose6d[step_index])

                ## is_first
                data = episode['is_first'][step_index]
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                h5_file.create_dataset(name='is_first', data=np_data)

                ## is_last
                data = episode['is_last'][step_index]
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                h5_file.create_dataset(name='is_last', data=np_data)

                ## is_terminal
                data = episode['is_terminal'][step_index]
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                h5_file.create_dataset(name='is_terminal', data=np_data)

    @staticmethod
    def dataset_info_to_dict(dataset_info: tfds.core.DatasetInfo) -> dict:
        info_dict = {
            'name': dataset_info.name,
            'version': str(dataset_info.version),
            'description': dataset_info.description,
            'homepage': dataset_info.homepage,
            'citation': dataset_info.citation,
            # 'splits': list(dataset_info.splits.keys()),
            'features': str(dataset_info.features),
        }
        return info_dict

    def _add_dict_to_group(self, h5_group, tar_dict: dict):
        for k, v in tar_dict.items():
            if isinstance(v, dict):
                g = h5_group.create_group(name=k)
                self._add_dict_to_group(g, v)
            else:
                try:
                    h5_group.create_dataset(name=k, data=v)
                except:
                    try:
                        h5_group.create_dataset(name=k, data=str(v))
                    except:
                        print(f'{k} can not be added into h5 file')

    def merge_shapes_dtypes(self, shapes, dtypes):
        assert shapes.keys() == dtypes.keys()

        res = dict()
        for k, v in shapes.items():
            if isinstance(v, dict):
                res[k] = self.merge_shapes_dtypes(shapes[k], dtypes[k])
            else:
                res[k] = {
                    'shape': shapes[k],
                    'dtype': dtypes[k]
                }
        return res

    def convert_origin_dataset_to_target(self, dataset, info, flag):
        # /dataset_0
        # |_meta_info.h5
        # |_/episodes
        # | |_/0
        # | | |_/steps
        # | |   |_/0
        # |     | |_other.h5
        # |     | |_XXX.jpg
        # |     |...
        # | |_/1
        # | |_...
        # /dataset_1
        # |
        episodes_dir = self.tgt_dir / 'episodes'
        episodes_dir.mkdir(exist_ok=True)
        num_episodes = int(dataset.cardinality())

        info_dict = self.dataset_info_to_dict(info)
        if self.rank == 0:
            with h5py.File(f'{str(self.tgt_dir)}/meta_info.h5', 'w') as h5_file:
                for k, v in info_dict.items():
                    h5_file.create_dataset(name=k, data=v)
                h5_file.create_dataset(name='num_episodes', data=num_episodes)

            with h5py.File(f'{str(self.tgt_dir)}/shape_info.h5', 'w') as h5_file:
                shapes = info.features.shape['steps']
                dtypes = info.features.dtype['steps']
                shape_types = self.merge_shapes_dtypes(shapes, dtypes)
                self._add_dict_to_group(h5_file, shape_types)

        for episode_index, episode in enumerate(tqdm.tqdm(dataset, total=num_episodes)):
            if episode_index < self.start_episode_idx:
                continue
            if self.end_episode_idx is not None:
                if episode_index >= self.end_episode_idx:
                    break
            if episode_index % self.num_worker != self.rank:
                continue

            self.process_episode(episode_dir=episodes_dir, episode=episode, episode_index=episode_index, flag=flag)
        dist.barrier()

    def run(self):
        builder = tfds.builder_from_directory(builder_dir=str(self.src_dir))
        dataset = dl.DLataset.from_rlds(
            builder, split='all', shuffle=False, num_parallel_reads=1
        )
        self.convert_origin_dataset_to_target(dataset, builder.info, self.flag)


def main(rank, port, num_worker, start_episode_idx=0, end_episode_idx=None, flag="success"):
    if num_worker > 1:
        setup(rank, world_size=num_worker, port=port)

    src_dir = f"/data/yifan26/LiuYuanzhe/Seer/datasets/droid_aff"  #
    tgt_dir = Path(f"/data/yifan26/LiuYuanzhe/Seer/datasets/droid_success_v2")
    tgt_dir.mkdir(exist_ok=True)

    dataset_converter = DatasetConverter(
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        rank=rank,
        num_worker=num_worker,
        start_episode_idx=start_episode_idx,  # the dataset[start_episode_idx] will be processed
        end_episode_idx=end_episode_idx,
        # None means the last episode. if not none, the dataset[end_episode_idx - 1] will be processed and the dataset[end_episode_idx] will not be processed
        flag=flag,
    )
    dataset_converter.run()


if __name__ == '__main__':
    start_episode_idx = 0
    end_episode_idx = 100000  # 100000
    num_worker = 4
    flag = "success"
    port = (random.randint(0, 3000) % 3000) + 27000

    assert num_worker > 1
    mp.spawn(main, args=(port, num_worker, start_episode_idx, end_episode_idx, flag), nprocs=num_worker, join=True)