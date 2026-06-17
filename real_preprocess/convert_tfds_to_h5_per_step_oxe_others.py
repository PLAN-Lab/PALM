import os
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import tqdm
import argparse
from pathlib import Path
import yaml
from PIL import Image
from octo_oxe_data_utils.dataset import make_single_dataset
from octo_oxe_data_utils.oxe import make_oxe_dataset_kwargs
import imageio
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
import math


tf.config.set_visible_devices([], "GPU")

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

def transform_action_jaco_play(actions):
    actions[:, 0], actions[:, 1] = actions[:, 1], actions[:, 0]
    actions *= np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    return actions

def transform_action_berkeley_autolab_ur5(action):
    action[:, 0], action[:, 1] = action[:, 1], action[:, 0]
    action *= np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])

    return action

def transform_action_viola(action):

    return action

def transform_action_taco_play(action):
    action[:3] /= 40  # cm -> m
    action[3:6] = action[3:6] / 20 #180 * np.pi  # degree -> rad

    return action

def transform_action_nyu_door_opening_surprising_effectiveness(action):
    return action

def transform_action_fractal20220817_data(actions):
    new_actions = np.zeros_like(actions)
    new_actions[:-1, 0:3] = (actions[1:, 0:3] - actions[:-1, 0:3]) * np.array([1.0, 1.0, 1.0])
    new_actions[:-1, 3:6] = (actions[1:, 3:6] - actions[:-1, 3:6]) * np.array([-1.0, -1.0, 1.0])

    return new_actions

def transform_action_bc_z(action):
    action[0], action[1] = action[1], action[0]
    action *= np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    action[3:6] = 0
    action[:3] /= 6
    action[3:6] /= 6
    return action

def transform_action_kuka(actions):
    actions *= np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0])

    return actions

def transform_action_bridge_dataset(actions):
    actions *= np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])

    return actions

def transform_action_roboturk(actions):
    actions *= np.array([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])

    return actions

def transform_action_dlr_edan_shared_control_converted_externally_to_rlds(action):
    new_action = np.zeros_like(action)
    new_action[:, 0] = action[:, 0] 
    new_action[:, 1] = action[:, 1] 
    new_action[:, 2] = action[:, 2] 
    zxy_rot = R.from_euler("zxy", action[:, 3:6])
    new_action[:, 3:6] = zxy_rot.as_euler("xyz")

    new_axis_action = np.zeros_like(action)
    new_axis_action[:, 0] = action[:, 1]
    new_axis_action[:, 1] = -action[:, 0]
    new_axis_action[:, 2] = action[:, 2]

    new_axis_rot = R.from_euler("xyz", [0.0, 0.0, -math.pi / 2]).as_matrix()
    zxy_rot = zxy_rot.as_matrix()
    result = np.zeros_like(zxy_rot)  
    for i in range(zxy_rot.shape[0]):
        result[i] = np.dot(new_axis_rot, zxy_rot[i])

    new_axis_action[:, 3:6] = R.from_matrix(result).as_euler("xyz")

    new_axis_action[:-1, :6] = (new_axis_action[1:, :6] - new_axis_action[:-1, :6]) 
    new_axis_action[-1] *= 0.0

    return new_axis_action

def transform_action_droid(action):

    return action

def transform_action_stanford_hydra_dataset_converted_externally_to_rlds(action):

    return action

def transform_action_austin_buds_dataset_converted_externally_to_rlds(action):
    action[0:3] *= 0.01

    return action

def transform_action_iamlab_cmu_pickup_insert_converted_externally_to_rlds(action):

    return action

def transform_action_utaustin_mutex(action):
    action[0:3] = action[0:3] * 0.01
    action[3:6] = action[3:6] * 0.05

    return action

def transform_action_language_table(action):

    return action

def transform_action_ucsd_kitchen_dataset_converted_externally_to_rlds(actions):
    new_action = np.zeros_like(actions)
    new_action[:-1, 0:3] = (actions[1:, 0:3] - actions[:-1, 0:3]) * 0.001
    new_action[:-1, 3:6] = (actions[1:, 3:6] - actions[:-1, 3:6]) / 180 * math.pi

    return new_action

def transform_action_cmu_stretch(action):

    return action

def transform_action_berkeley_fanuc_manipulation(action):

    return action

transform_action = {
    'jaco_play': transform_action_jaco_play,
    'berkeley_autolab_ur5': transform_action_berkeley_autolab_ur5,
    'fractal20220817_data': transform_action_fractal20220817_data,
    'kuka': transform_action_kuka,
    'bridge_dataset': transform_action_bridge_dataset,
    'roboturk': transform_action_roboturk,
    'dlr_edan_shared_control_converted_externally_to_rlds': transform_action_dlr_edan_shared_control_converted_externally_to_rlds,
    'language_table': transform_action_language_table,
    'ucsd_kitchen_dataset_converted_externally_to_rlds': transform_action_ucsd_kitchen_dataset_converted_externally_to_rlds,
    'cmu_stretch': transform_action_cmu_stretch,
    'berkeley_fanuc_manipulation': transform_action_berkeley_fanuc_manipulation
}

class DatasetConverter:
    def __init__(
        self,
        dataset_name: str,
        src_root_dir,
        tgt_dataset_dir: Path,
        rank: int,
        num_worker: int,
    ):
        self.dataset_name = dataset_name
        self.src_root_dir = src_root_dir
        self.tgt_dataset_dir = tgt_dataset_dir
        self.rank = rank
        self.num_worker = num_worker

    def process_episode(self, episode_dir, episode):
        # get episode length
        num_steps = episode['action'].shape[0]

        # get language instruction
        language_instructions = episode['task']['language_instruction']

        # save episode length
        with h5py.File(f'{episode_dir}/meta_info.h5', 'w') as h5_file:
            h5_file.create_dataset(name='length', data=num_steps)

        actions = episode['action']
        actions = np.array(actions).astype(to_np_dtype(actions.dtype))
        actions = transform_action[self.dataset_name](actions)

        steps_dir = episode_dir/'steps'
        steps_dir.mkdir(exist_ok=True)
        for step_index in range(num_steps):
            step_dir = episode_dir/'steps'/str(step_index).zfill(4)
            step_dir.mkdir(exist_ok=True)
            with h5py.File(f'{step_dir}/other.h5', 'w') as h5_file:
                # language instruction
                h5_file.create_dataset('language_instruction', data=np.array(language_instructions[step_index], dtype=h5py.string_dtype(encoding='utf-8')))

                # episode length
                h5_file.create_dataset(name='episode_length', data=num_steps)

                # timestep
                h5_file.create_dataset(name='timestep', data=step_index)

                # action
                if actions[step_index, 6] == 0.0:
                    actions[step_index, 6] = -1.0
                h5_file.create_dataset(name='action_delta_tcp_pose', data=actions[step_index])

                observation_group = h5_file.create_group(name='observation')
                for data_key in episode['observation'].keys():
                    data = episode['observation'][data_key][step_index]
                    if data.dtype == tf.string:  # decode image
                        try:
                            data = tf.io.decode_image(data, expand_animations=False, dtype=tf.uint8)
                        except:
                            if not 'image' in data_key:
                                import pdb;pdb.set_trace()
                            print('padding image')
                            data = tf.zeros([1, 1, 3], dtype=tf.uint8)

                    np_data = np.array(data).astype(to_np_dtype(data.dtype))
                    if len(np_data.shape) == 3:  # image
                        Image.fromarray(np_data).save(f'{step_dir}/{data_key}.jpg')
                    else:
                        observation_group.create_dataset(name=data_key, data=np_data)
                
                # is_first
                data = episode['is_first'][step_index]
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                h5_file.create_dataset(name='is_first', data=np_data)

                # is_last
                data = episode['is_last'][step_index]
                np_data = np.array(data).astype(to_np_dtype(data.dtype))
                h5_file.create_dataset(name='is_last', data=np_data)

                # is_terminal
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
                    # print(f'try to convert data of {k} to string')
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

    def convert_origin_dataset_to_target(self, target_dir, merged_dataset, info):
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

        episodes_dir = self.tgt_dataset_dir/'episodes'
        episodes_dir.mkdir(exist_ok=True)
        num_episodes = int(merged_dataset.cardinality())

        info_dict = self.dataset_info_to_dict(info)
        if self.rank == 0:
            with h5py.File(f'{target_dir}/meta_info.h5', 'w') as h5_file:
                for k, v in info_dict.items():
                    h5_file.create_dataset(name=k, data=v)
                h5_file.create_dataset(name='num_episodes', data=num_episodes)
    
            with h5py.File(f'{target_dir}/shape_info.h5', 'w') as h5_file:
                shapes = info.features.shape['steps']
                dtypes = info.features.dtype['steps']
                shape_types = self.merge_shapes_dtypes(shapes, dtypes)
                self._add_dict_to_group(h5_file, shape_types)

        for episode_index, episode in enumerate(tqdm.tqdm(merged_dataset, total=num_episodes)):
            if episode_index % self.num_worker != self.rank:
                continue
            episode_dir = episodes_dir/str(episode_index).zfill(6)
            episode_dir.mkdir(exist_ok=True)
            self.process_episode(episode_dir=episode_dir, episode=episode)

    def run(self):
        print(f'target root dir: {self.tgt_dataset_dir}')

        dataset_kwargs = make_oxe_dataset_kwargs(
            self.dataset_name,
            self.src_root_dir,
        )
        merged_dataset, info = make_single_dataset(dataset_kwargs)
        
        self.convert_origin_dataset_to_target(self.tgt_dataset_dir, merged_dataset, info)

        print(f'data saved at {self.tgt_dataset_dir}')

def main(rank, port, num_worker):
    if num_worker > 1:
        setup(rank, world_size=num_worker, port=port)

    dataset_names = [
        "berkeley_autolab_ur5",
        "jaco_play",
        "roboturk",
        "ucsd_kitchen_dataset_converted_externally_to_rlds",
        "berkeley_fanuc_manipulation",
        "cmu_stretch",
        "bridge_dataset",
        "language_table",
        "fractal20220817_data",
        "kuka",
        "dlr_edan_shared_control_converted_externally_to_rlds",
        "nyu_door_opening_surprising_effectiveness",
        "bc_z",
    ]

    for dataset_name in dataset_names:
        src_root_dir = 'your_path_to_open_x_embodiment' #'s3://open_x_embodiment' 
        tgt_dataset_dir = Path('oxe')/dataset_name
        tgt_dataset_dir.mkdir(exist_ok=True) 

        dataset_converter = DatasetConverter(
            dataset_name=dataset_name,
            src_root_dir=src_root_dir,
            tgt_dataset_dir=tgt_dataset_dir,
            rank=rank,
            num_worker=num_worker,
        )
        dataset_converter.run()
        dist.barrier()

if __name__ == '__main__':
    num_worker = 4
    port = (random.randint(0, 3000) % 3000) + 27000
    mp.spawn(main, args=(port, num_worker), nprocs=num_worker, join=True)
