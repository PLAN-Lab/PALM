import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import numpy as np
import wandb
import functools
import clip
import torch
from PIL import Image as PILImage
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
from torch.nn.parallel import DistributedDataParallel as DDP
from models.palm_model import PALMAgent
from utils.distributed_utils import init_distributed_device, world_info_from_env
from utils.arguments_utils import get_parser
from utils.train_utils import get_cast_dtype
from utils.data_utils import preprocess_image, preprocess_text_calvin


class PALMController:
    def __init__(self):
        super().__init__()
        parser = get_parser(is_eval=True)
        args = parser.parse_args()
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        device_id = init_distributed_device(args)
        args.device_id = device_id
        self.random_seed(args.seed)
        self.args = args  
        self.device_id = args.device_id

        # setup model
        self.setup_model()

        # setup inference wrapper
        self.cast_dtype = get_cast_dtype(self.args.precision)
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=clip)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=self.model.image_processor)
        self.action_hist_queue = []
        self.history_len = self.args.sequence_length
        self.action_pred_steps = self.args.action_pred_steps
        self.use_ensembling =self.args.eval_libero_ensembling
        self.ensembling_temp = self.args.ensembling_temp
        self.gripper_width = self.args.gripper_width
        self.real_eval_max_steps = self.args.real_eval_max_steps
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)
        self.cnt = 0
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.real_eval_max_steps,
                        self.real_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device_id)
        
    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)
        self.gripper_state = np.array([-1.0])
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.real_eval_max_steps,
                        self.real_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device_id)
        self.cnt += 1

    def random_seed(self, seed=42, rank=0):
        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + rank)
            torch.cuda.manual_seed_all(seed + rank)  # if you are using multi-GPU.
        np.random.seed(seed + rank)  # Numpy module.
        random.seed(seed + rank)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def setup_model(self):
        self.model = PALMAgent(
            finetune_type=self.args.finetune_type,
            clip_device=self.device_id,
            vit_checkpoint_path=self.args.vit_checkpoint_path,
            sequence_length=self.args.sequence_length,
            num_resampler_query=self.args.num_resampler_query,
            num_obs_token_per_image=self.args.num_obs_token_per_image,
            calvin_input_image_size=self.args.calvin_input_image_size,
            patch_size=self.args.patch_size,
            action_pred_steps=self.args.action_pred_steps,
            obs_pred=self.args.obs_pred,
            atten_only_obs=self.args.atten_only_obs,
            attn_robot_proprio_state=self.args.attn_robot_proprio_state,
            atten_goal=self.args.atten_goal,
            atten_goal_state=self.args.atten_goal_state,
            mask_l_obs_ratio=self.args.mask_l_obs_ratio,
            transformer_layers=self.args.transformer_layers,
            hidden_dim=self.args.hidden_dim,
            transformer_heads=self.args.transformer_heads,
            phase=self.args.phase,
            gripper_width=self.args.gripper_width,
        )

        # bf16 or fp32
        if self.args.precision == "bf16" or self.args.precision == "amp_bfloat16" or self.args.precision == "amp_bf16":
            self.model = self.model.bfloat16()
        elif self.args.precision == "fp16":
            self.model = self.model.half()
        elif self.args.precision == "fp32":
            self.model = self.model.float()
            if 'vision_encoder' in self.args.bf16_module:
                self.model.vision_encoder.bfloat16()
            if "causal_transformer" in self.args.bf16_module:
                self.model.transformer_backbone.bfloat16()
            if "image_decoder" in self.args.bf16_module:
                self.model.image_decoder.bfloat16()
                self.model.image_decoder_obs_pred_projector.bfloat16()

        # regularize model's gradients
        self.model.clip_model.requires_grad_(False)
        self.model.vision_encoder.requires_grad_(False)
        self.model = self.model.to(self.device_id)
        self.model._init_model_type()

        # DDP
        self.ddp_model = DDP(self.model, device_ids=[self.device_id], find_unused_parameters=True)
        if self.args.resume_from_checkpoint is not None:
            if self.args.rank == 0:
                print(f"Loading checkpoint from {self.args.resume_from_checkpoint}")
            checkpoint = torch.load(self.args.resume_from_checkpoint, map_location="cpu")
            self.ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        self.ddp_model.eval()

    def forward(self, obs_dict, include_info=False, timestep=0):
        pass #TODO
        # preprocess image 
        image_x = obs_dict["color_image"][0]
        image_x = PILImage.fromarray(image_x).convert('RGB')
        image_x = self.image_process_fn([image_x])
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_dtype)

        gripper_x = obs_dict["color_image"][1]
        gripper_x = PILImage.fromarray(gripper_x).convert('RGB')
        gripper_x = self.image_process_fn([gripper_x])
        gripper_x = gripper_x.unsqueeze(1).to(dtype=self.cast_dtype)

        # preprocess text
        text_x = self.text_process_fn([obs_dict["language_instruction"]])
        text_x = text_x.unsqueeze(1)

        # preprocess state
        gripper_xyzeuler = obs_dict["robot_state"]["pose6d"]
        gripper_state = obs_dict["robot_state"]["gripper_open_state"]
        gripper_position = obs_dict["robot_state"]["gripper_position"]
        if not self.gripper_width:
            state_x = torch.from_numpy(np.concatenate([gripper_xyzeuler, gripper_state])).to(dtype=self.cast_dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        else:
            state_x = torch.from_numpy(np.concatenate([gripper_xyzeuler, gripper_position, gripper_position])).to(dtype=self.cast_dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, 8]
        
        with torch.no_grad():
            image_x = image_x.to(self.device_id)
            gripper_x = gripper_x.to(self.device_id)
            text_x = text_x.to(self.device_id)
            state_x = state_x.to(self.device_id)
            self.img_queue.append(image_x)  
            self.gripper_queue.append(gripper_x)
            self.state_queue.append(state_x)
            if len(self.text_queue) == 0 and text_x is not None:  
                self.text_queue.append(text_x)
                for _ in range(self.args.sequence_length - 1):
                    self.text_queue.append(text_x)

            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            num_step = image_primary.shape[1]
            if num_step < self.history_len:  # padding
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len-num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state

            arm_action, gripper_action, _, _, _ = self.ddp_model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=torch.zeros(1, self.history_len, 7).to(input_state.device),
            )
            if not self.use_ensembling:
                action = torch.concat((arm_action[0, :, 0, :], gripper_action[0, :, 0, :] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
                action = action.cpu().detach().numpy()
                if num_step < self.history_len:
                    action = action[num_step - 1]
                else:
                    action = action[-1]
            else:
                if num_step < self.history_len:
                    selected_step = num_step - 1
                else:
                    selected_step = -1
                action = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1) # (1, action_pred_steps, 7)
                self.all_time_actions[timestep:timestep+1,timestep:timestep+self.action_pred_steps] = action 
                actions_for_curr_step = self.all_time_actions[:, timestep]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = self.ensembling_temp
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device_id).unsqueeze(dim=1)
                action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
                action = action.detach().cpu().numpy()[-1]
        target_pos = action[:3]
        target_euler = action[3:6]
        target_gripper = action[6]
        is_terminal = -1.0

        return target_pos, target_euler, target_gripper, is_terminal

if __name__ == "__main__":
    controller = SeerController()
