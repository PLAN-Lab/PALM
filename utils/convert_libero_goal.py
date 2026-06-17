import os
import re
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import h5py
import tqdm
import argparse
from pathlib import Path
import yaml
from PIL import Image
import time
import json
import math
from scipy.spatial.transform import Rotation


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def extract_task_information(file_name, path):
    """
    Extracts task information from the given file name.
    For libero_goal, the file names are long descriptive names like:
    'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5'
    """
    # Get the base filename without path
    basename = os.path.basename(file_name)
    # Remove '_demo.hdf5' suffix
    task_name = basename.replace('_demo.hdf5', '')
    # Convert underscores to spaces for task language
    task_language = task_name.replace('_', ' ')
    
    print(f"Task: {task_language}")
    return task_name, task_language


class DatasetConverter:
    def __init__(
        self,
        src_dir: str,
        tgt_dir: str,
        rank: int,
        num_worker: int,
        start_episode_idx,
        end_episode_idx,
    ):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.rank = rank
        self.num_worker = num_worker
        self.start_episode_idx = start_episode_idx
        self.end_episode_idx = end_episode_idx

    def process_episode(self, episode_dir, language_instructions, demo_data, episode_index, episode_index_in_task):
        i = episode_index_in_task
        # get episode dir
        episode_dir.mkdir(exist_ok=True)

        # Get agent's view camera
        obs = np.array(demo_data['demo_{}'.format(i)]['obs']['agentview_rgb'])
        
        # Get wrist's view camera
        obs_wrist = np.array(demo_data['demo_{}'.format(i)]['obs']['eye_in_hand_rgb'])

        # Get actions
        action = np.array(demo_data['demo_{}'.format(i)]['actions'])  # -1 open, 1 close
        
        joint_state = np.array(demo_data['demo_{}'.format(i)]['obs']['joint_states'])
        ee_pos = np.array(demo_data['demo_{}'.format(i)]['obs']['ee_pos'])
        ee_ori = np.array(demo_data['demo_{}'.format(i)]['obs']['ee_ori'])
        ee_state = np.array(demo_data['demo_{}'.format(i)]['obs']['ee_states'])

        gripper_state = np.zeros_like(action[:, -1])
        gripper_state[1:] = action[:-1, -1]
        gripper_state[0] = action[0, -1]

        gripper_position = np.array(demo_data['demo_{}'.format(i)]['obs']['gripper_states'])
        gripper_command = action[:, -1]

        # get episode length
        num_steps = obs.shape[0]
        
        episode_dir = episode_dir/str(episode_index).zfill(6)
        episode_dir.mkdir(exist_ok=True)

        # save episode length and language instruction
        with h5py.File(f'{episode_dir}/meta_info.h5', 'w') as h5_file:
            h5_file.create_dataset(name='length', data=num_steps)
        
        steps_dir = episode_dir/'steps'
        steps_dir.mkdir(exist_ok=True)
        for step_index in range(num_steps):
            step_dir = episode_dir/'steps'/str(step_index).zfill(4)
            step_dir.mkdir(exist_ok=True)
            
            with h5py.File(f'{step_dir}/other.h5', 'w') as h5_file:
                # language instruction
                h5_file.create_dataset('language_instruction', data=np.array(language_instructions, dtype=h5py.string_dtype(encoding='utf-8')))

                # episode length
                h5_file.create_dataset(name='episode_length', data=num_steps)

                # action
                h5_file.create_dataset(name='action', data=action[step_index])

                # observation (timestep, proprio, image_XXX)
                observation_group = h5_file.create_group(name='observation')

                # Save images
                Image.fromarray(obs[step_index]).save(f'{step_dir}/image_primary.jpg')
                Image.fromarray(obs_wrist[step_index]).save(f'{step_dir}/image_wrist.jpg')

                # proprio
                observation_group.create_dataset(name='proprio', data=joint_state[step_index])

                # tcp_pose
                observation_group.create_dataset(name='tcp_pose', data=ee_state[step_index])

                # gripper state (-1 or 1)
                observation_group.create_dataset(name='gripper_state', data=gripper_state[step_index])

                # gripper position (n, 2)
                observation_group.create_dataset(name='gripper_position', data=gripper_position[step_index])

    def convert_origin_dataset_to_target(self, dataset_by_task):
        episodes_dir = self.tgt_dir/'episodes'
        episodes_dir.mkdir(exist_ok=True)

        num_episodes = 0
        for dataset in dataset_by_task:
            num_episodes += dataset['num_episode']

        if self.rank == 0:
            with h5py.File(f'{str(self.tgt_dir)}/meta_info.h5', 'w') as h5_file:
                h5_file.create_dataset(name='num_episodes', data=num_episodes)

        processed_task_num_episode = 0
        task_index = 0

        for episode_index in range(num_episodes):
            episode_index_in_task = episode_index - processed_task_num_episode

            if episode_index < self.start_episode_idx:
                continue
            if self.end_episode_idx is not None:
                if episode_index >= self.end_episode_idx:
                    break
            if episode_index % self.num_worker != self.rank:
                if episode_index_in_task+1 == dataset_by_task[task_index]['num_episode']:
                    processed_task_num_episode += dataset_by_task[task_index]['num_episode']
                    task_index += 1
                continue
            print(self.rank, episode_index, '/' , num_episodes)
            self.process_episode(episode_dir=episodes_dir, language_instructions=dataset_by_task[task_index]['language'], demo_data=dataset_by_task[task_index]['data'], episode_index=episode_index, episode_index_in_task=episode_index_in_task)

            if episode_index_in_task+1 == dataset_by_task[task_index]['num_episode']:
                processed_task_num_episode += dataset_by_task[task_index]['num_episode']
                task_index += 1

    def run(self):
        print(f'target dir: {self.tgt_dir}')

        dataset_by_task = []
        # Sort files for consistent ordering
        file_paths = sorted(list(Path(self.src_dir).iterdir()))
        for path in file_paths:
            path_name = str(path)
            if not path_name.endswith('.hdf5'):
                continue
            task_name, task_language = extract_task_information(path_name, self.src_dir)
            demo_data = h5py.File(path_name, 'r')['data']
            num_episode = len(demo_data)
            dataset = {
                'language': task_language,
                'num_episode': num_episode,
                'data': demo_data
            }
            dataset_by_task.append(dataset)

        self.convert_origin_dataset_to_target(dataset_by_task)

        print(f'data saved at {self.tgt_dir}')

        # get data_info.json
        data_info = []
        episode_idx = 0
        total_step = 0
        
        # Sort files again for consistent episode ordering
        file_paths = sorted(list(Path(self.src_dir).iterdir()))
        for path in file_paths:
            path_name = str(path)
            if not path_name.endswith('.hdf5'):
                continue
            demo_data = h5py.File(path_name, 'r')['data']
            num_episode = len(demo_data)
            for i in range(num_episode):
                num_steps = np.array(demo_data['demo_{}'.format(i)]['obs']['agentview_rgb']).shape[0]
                data_info.append([str(episode_idx).zfill(6), num_steps])
                episode_idx += 1
                total_step += num_steps
        
        print(f'Total episodes: {episode_idx}, Total steps: {total_step}')
        
        # Save to data_info directory
        data_info_dir = Path('./data_info')
        data_info_dir.mkdir(exist_ok=True)
        with open(f'./data_info/{dataset_name}_converted.json', 'w') as f:
            json.dump(data_info, f)
        
        print(f'Data info saved to ./data_info/{dataset_name}_converted.json')


def main(rank, port, num_worker, start_episode_idx=0, end_episode_idx=None):
    if num_worker > 1:
        setup(rank, world_size=num_worker, port=port)

    global dataset_name
    dataset_name = "libero_goal"
    src_dir = "/data/yifan26/LiuYuanzhe/LIBERO/datasets/datasets/libero_goal"
    tgt_dir = Path(f"/data/yifan26/LiuYuanzhe/Seer/{dataset_name}_converted")
    tgt_dir.mkdir(exist_ok=True) 

    dataset_converter = DatasetConverter(
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        rank=rank,
        num_worker=num_worker,
        start_episode_idx=start_episode_idx,
        end_episode_idx=end_episode_idx,
    )
    dataset_converter.run()


if __name__ == '__main__':
    start_episode_idx = 0
    end_episode_idx = None
    num_worker = 8
    port = (random.randint(0, 3000) % 3000) + 27000

    assert num_worker > 1
    mp.spawn(main, args=(port, num_worker, start_episode_idx, end_episode_idx), nprocs=num_worker, join=True)

