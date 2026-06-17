"""Open X-Embodiment Dataset Transforms

input: dict of features, each is batched, i.e. has leading time dimension
expected output:
step = {
    'observation': {
        <image_keys, depth_image_keys>
        state in chosen state representation
    },
    'action': action in chosen action representation,
    'language_instruction': str,
}
"""

from typing import Any, Dict

import tensorflow as tf
import numpy as np
from octo_oxe_data_utils.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_actions,
)
from pdb import set_trace
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation

def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft
    # make gripper action absolute action, +1 = open, 0 = close

    # set_trace()
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"]["world_vector"],
    #         trajectory["action"]["rotation_delta"],
    #         gripper_action[:, None],
    #     ),
    #     axis=-1,
    # )

    trajectory["action"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["base_pose_tool_reached"][:, 3:7]),
            gripper_action[:, None],
        ),
        axis=-1,
    )



    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"][
        "clip_function_input/base_pose_tool_reached"
    ] = tf.reshape(eef_value, (-1, 7))
    gripper_value = tf.io.decode_compressed(
        trajectory["observation"]["gripper_closed"], compression_type="ZLIB"
    )
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1, 1))
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def taco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["robot_obs"][
        :, :6
    ]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["robot_obs"][
        :, 7:8
    ]
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"][
        "end_effector_cartesian_pos"
    ][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"][
        "end_effector_cartesian_pos"
    ][:, -1:]

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def berkeley_cable_routing_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(
        tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1)
    )

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )

    
    # joint_states = trajectory["observation"]["joint_states"] # N x 7
    
    # print("joint_states.shape: ", joint_states.shape)
    # print("joint_states: ", joint_states)
    # print("world_vector: ", trajectory["action"]["rotation_delta"])
    # set_trace()
    # robot = rtb.models.Panda()
    # joint_states = tf.compat.v1.Session().run(joint_states)
    # print("joint_states: ", joint_states)
    # tcp_poses = robot.fkine(joint_states[0])
    # tcp_poses = [tcp_poses[i].data for i in range(tf.shape(joint_states)[0])]
    # tcp_poses = np.concatenate(tcp_poses,axis=0) # N x 4 x 4
    # tcp_translations = tcp_poses[:, :3, 3] # N x 3
    # tcp_eulers = Rotation.from_matrix(tcp_poses[:, :3, :3]).as_euler("xyz") # N x 3

    # trajectory["action"] = tf.concat(
    #     (
    #         tf.convert_to_tensor(tcp_translations),
    #         tf.convert_to_tensor(tcp_eulers),
    #         gripper_action,
    #     ),
    #     axis=-1,
    # )


    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def berkeley_autolab_ur5_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][
        :, 6:14
    ]
    trajectory["observation"]["depth"] = trajectory["observation"].pop(
        "image_with_depth"
    )

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(
        instruction_bytes, output_encoding="UTF-8"
    )
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[
        :, :1
    ].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][
        ..., 0
    ]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        ..., -1:
    ]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"][:, :6],
    #         invert_gripper_actions(trajectory["action"][:, -1:]),
    #     ),
    #     axis=-1,
    # )
    
    new_trans = tf.concat(
        (
            
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            tf.zeros_like(trajectory["observation"]["state"][:1, :3]),
            # trajectory["observation"]["state"][:, :3],
        ),
        axis=0,
    )

    new_rot = tf.concat(
        (
            
            trajectory["observation"]["state"][1:, 7:10] - trajectory["observation"]["state"][:-1, 7:10],
            tf.zeros_like(trajectory["observation"]["state"][:1, 7:10]),
            # trajectory["observation"]
        ),
        axis=0,
    )

    trajectory["action"] = tf.concat(
        (
            # new_trans,
            # new_rot,
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, 10:17]


    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -3:-2
    ]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    import tensorflow_graphics.geometry.transformation as tft
    gripper_pose = tf.reshape(trajectory["observation"]["state"][:, 8:], (-1, 4, 4))
    position = gripper_pose[:, :3, 3]
    euler_angles = tft.euler.from_rotation_matrix(gripper_pose[:, :3, :3])
    print("position.shape: ", position.shape)
    print("euler_angles.shape: ", euler_angles.shape)

    trajectory["action"] = tf.concat(
        (
            # trajectory["action"][:, :6],
            # tf.expand_dims(position, axis=0),
            # tf.expand_dims(euler_angles, axis=0),
            position,
            euler_angles,
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )

    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(
        trajectory["observation"]["depth"][..., 0], tf.float32
    )
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        ..., 7:8
    ]
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # set_trace()
    print("keys in present: ", trajectory["observation"].keys())
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(
                tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)
            ),
        ),
        axis=-1,
    )

    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["observation"]["present/xyz"][:, :3],
    #         trajectory["observation"]["present/axis_angle"][:, :3],
    #         invert_gripper_actions(
    #             tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)
    #         ),
    #     ),
    #     axis=-1,
    # )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def kaist_nonprehensible_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"][
        "end_effector_pose"
    ][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, 7:8
    ]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, 6:7
    ]

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["gripper_state"]),
        ),
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
}


DATASET_TRANSFORMS=(
    "fractal20220817_data 0.1.0 resize_and_jpeg_encode"  # EEFP # #
    "bridge 0.1.0 resize_and_jpeg_encode"  # EEFP ##
    "kuka 0.1.0 resize_and_jpeg_encode,filter_success"  # EEFP #
    "taco_play 0.1.0 resize_and_jpeg_encode"  # EEFP ##
    "jaco_play 0.1.0 resize_and_jpeg_encode"  # EEFP #
    "berkeley_cable_routing 0.1.0 resize_and_jpeg_encode"  # EEFV #
    "roboturk 0.1.0 resize_and_jpeg_encode"  # EEFP ##
    "nyu_door_opening_surprising_effectiveness 0.1.0 resize_and_jpeg_encode"  # EEFP （black image
    "viola 0.1.0 resize_and_jpeg_encode"  # EEFP #
    "berkeley_autolab_ur5 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels"  # EEFP ##
    "toto 0.1.0 resize_and_jpeg_encode"  # Joint Position #
    "language_table 0.1.0 resize_and_jpeg_encode"  # EEFP #
    "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"  # EEFP #
    "austin_buds_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFP ## TODO
    "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFV #
    "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFV #
    "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFP ##
    "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFV #
    "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFV #
    "bc_z 1.0.0 resize_and_jpeg_encode"  # EEFP #
    "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFP ##
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"  # EEFP #
    "utaustin_mutex 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"  # EEFP ##
    "berkeley_fanuc_manipulation 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"  # EEFP ##
    "cmu_stretch 0.1.0 resize_and_jpeg_encode"  # EEFP ##
)

OXE_MAGIC_SOUP = [
    ("fractal20220817_data", 0.54087122203), # EEFP #
    ("kuka", 0.8341046294), # EEFP #
    ("bridge_dataset", 1.0), # EEFP #
    ("taco_play", 2.0), # EEFP #
    ("jaco_play", 1.0), # EEFP #
    ("berkeley_cable_routing", 1.0), # EEFV #
    ("roboturk", 2.0), # EEFP ##
    ("nyu_door_opening_surprising_effectiveness", 1.0), # EEFP （black image
    ("viola", 2.0), # EEFP #
    ("berkeley_autolab_ur5", 2.0), # EEFP #
    ("toto", 1.0), # Joint Position #
    ("language_table", 0.1),  # EEFP #
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0), # EEFP #
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0), # EEFP #
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0), # EEFV #
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1), # EEFV #
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0), # EEFP #
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0), # EEFV #
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0), # EEFV #
    ("bc_z", 0.2),  # EEFP #
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0), # EEFP #
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0), # EEFP #
    ("utaustin_mutex", 1.0),  # EEFP #
    ("berkeley_fanuc_manipulation", 2.0), # EEFP #
    ("cmu_stretch", 1.0), # EEFP #
]

EEFP_OCTO_SOUP = [
    ("fractal20220817_data", 1.0), # EEFP (world_vector + rotation_delta) #
    ("kuka", 1.0), # EEFP (world_vector + rotation_delta) #
    ("bridge_dataset", 1.0), # EEFP #
    ("taco_play", 1.0), # EEFP #
    ("jaco_play", 1.0), # EEFP #
    ("roboturk", 1.0), # EEFP ##
    ("nyu_door_opening_surprising_effectiveness", 1.0), # EEFP （black image
    ("viola", 1.0), # EEFP #
    ("berkeley_autolab_ur5", 1.0), # EEFP # 

    
    ("language_table", 1.0),  # EEFP #
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0), # EEFP #
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0), # EEFP #
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 1.0), # EEFP #
    ("bc_z", 1.0),  # EEFP #
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0), # EEFP #
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0), # EEFP #
    ("utaustin_mutex", 1.0),  # EEFP #
    ("berkeley_fanuc_manipulation", 1.0), # EEFP #
    ("cmu_stretch", 1.0), # EEFP #
    # action is 
]