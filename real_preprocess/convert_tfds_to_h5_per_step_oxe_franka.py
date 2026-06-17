import time
import os
import random
import time
import importlib_resources
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

def transformation_matrix_to_trans_and_euler(transformation_matrix):
    translation = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)

    return translation, euler

robot = rtb.models.Panda()

def franka_joint_position_to_tcp_pose(joint_position):
    Te = robot.fkine(joint_position)  # forward kinematics
    transformation_matrix = np.array(Te)
    translation, euler = transformation_matrix_to_trans_and_euler(transformation_matrix)

    return np.concatenate([translation, euler])

def franka_joint_position_to_wrist_pose(joint_position):
    end_link = robot.link_dict['panda_hand']
    Te = robot.fkine(joint_position, end=end_link)  # forward kinematics
    transformation_matrix = np.array(Te)
    translation, euler = transformation_matrix_to_trans_and_euler(transformation_matrix)

    return np.concatenate([translation, euler])

def convert_franka_action(actions, joint_positions):
    action_tcp_poses, action_wrist_poses, action_delta_tcp_poses, action_delta_wrist_poses = actions.copy(), actions.copy(), actions.copy(), actions.copy()
    tcp_pose_current = franka_joint_position_to_tcp_pose(joint_positions[0])
    wrist_pose_current = franka_joint_position_to_wrist_pose(joint_positions[0])

    for idx in range(actions.shape[0]):
        if actions[idx, 6] == 0.0:
            action_tcp_poses[idx, 6], action_wrist_poses[idx, 6], action_delta_tcp_poses[idx, 6], action_delta_wrist_poses[idx, 6] = -1.0, -1.0, -1.0, -1.0
        if idx == actions.shape[0] - 1:  # the last step, padding
            action_tcp_poses[idx, :6] = tcp_pose_current
            action_wrist_poses[idx, :6] = wrist_pose_current
            action_delta_tcp_poses[idx, :6] = np.zeros_like(tcp_pose_current)
            action_delta_wrist_poses[idx, :6] = np.zeros_like(wrist_pose_current)
            break

        tcp_pose_next = franka_joint_position_to_tcp_pose(joint_positions[idx+1])
        wrist_pose_next = franka_joint_position_to_wrist_pose(joint_positions[idx+1])

        action_tcp_poses[idx, :6] = tcp_pose_next
        action_wrist_poses[idx, :6] = wrist_pose_next
        action_delta_tcp_poses[idx, :6] = tcp_pose_next - tcp_pose_current
        action_delta_wrist_poses[idx, :6] = wrist_pose_next - wrist_pose_current

        tcp_pose_current = tcp_pose_next
        wrist_pose_current = wrist_pose_next

    return action_tcp_poses, action_wrist_poses, action_delta_tcp_poses, action_delta_wrist_poses

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
        joint_positions = episode['observation']['proprio']
        joint_positions = np.array(joint_positions).astype(to_np_dtype(joint_positions.dtype))
        action_tcp_poses, action_wrist_poses, action_delta_tcp_poses, action_delta_wrist_poses = convert_franka_action(actions, joint_positions)
        
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
                h5_file.create_dataset(name='action_delta_tcp_pose', data=action_delta_tcp_poses[step_index])
                h5_file.create_dataset(name='action_delta_wrist_pose', data=action_delta_wrist_poses[step_index])
                h5_file.create_dataset(name='action_tcp_pose', data=action_tcp_poses[step_index])
                h5_file.create_dataset(name='action_wrist_pose', data=action_wrist_poses[step_index])

                # observation (timestep, proprio, image_XXX)
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
        import pdb;pdb.set_trace()

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
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "viola",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "austin_buds_dataset_converted_externally_to_rlds",
        "utaustin_mutex",
        "taco_play",
    ]

    for dataset_name in dataset_names:
        src_root_dir = 'your_path_to_open_x_embodiment'  #'s3://open_x_embodiment' 
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