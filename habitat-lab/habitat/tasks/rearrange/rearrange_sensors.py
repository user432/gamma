#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    CollisionDetails,
    UsesArticulatedAgentInterface,
    batch_transform_point,
    get_angle_to_pos,
    rearrange_logger,
)
from habitat.tasks.utils import cartesian_to_polar
import magnum as mn


from graspness.models.graspnet import GraspNet, pred_decode
from graspness.dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
from graspness.utils.collision_detector import ModelFreeCollisionDetector
from graspness.utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask

from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

import cv2

from pfusion.octree import GL_tree
from pfusion.octree_grasp import GL_tree_grasp
from scipy.spatial.transform import Rotation
import MinkowskiEngine as ME
import collections.abc as container_abcs

# import scipy.io as scio

import torch
import time


class MultiObjSensor(PointGoalSensor):
    """
    Abstract parent class for a sensor that specifies the locations of all targets.
    """

    def __init__(self, *args, task, **kwargs):
        self._task = task
        self._sim: RearrangeSim
        super().__init__(*args, task=task, **kwargs)

    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(n_targets * 3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_sensor
class TargetCurrentSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    This is the ground truth object position sensor relative to the robot end-effector coordinate frame.
    """

    cls_uuid: str = "obj_goal_pos_sensor"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        self._sim: RearrangeSim
        T_inv = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .inverted()
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        pos = scene_pos[idxs]

        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])

        return pos.reshape(-1)


@registry.register_sensor
class TargetStartSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "obj_start_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()
        pos = self._sim.get_target_objs_start()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


class PositionGpsCompassSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        n_targets = self._task.get_n_targets()
        self._polar_pos = np.zeros(n_targets * 2, dtype=np.float32)
        return spaces.Box(
            shape=(n_targets * 2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_positions(self) -> np.ndarray:
        raise NotImplementedError("Must override _get_positions")

    def get_observation(self, task, *args, **kwargs):
        pos = self._get_positions()
        articulated_agent_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation

        rel_pos = batch_transform_point(
            pos, articulated_agent_T.inverted(), np.float32
        )

        for i, rel_obj_pos in enumerate(rel_pos):
            rho, phi = cartesian_to_polar(rel_obj_pos[0], rel_obj_pos[1])
            self._polar_pos[(i * 2) : (i * 2) + 2] = [rho, -phi]

        return self._polar_pos


@registry.register_sensor
class TargetStartGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_start_gps_compass"

    def _get_uuid(self, *args, **kwargs):
        return TargetStartGpsCompassSensor.cls_uuid

    def _get_positions(self) -> np.ndarray:
        return self._sim.get_target_objs_start()


@registry.register_sensor
class TargetGoalGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_goal_gps_compass"

    def _get_uuid(self, *args, **kwargs):
        return TargetGoalGpsCompassSensor.cls_uuid

    def _get_positions(self) -> np.ndarray:
        _, pos = self._sim.get_targets()
        return pos


@registry.register_sensor
class AbsTargetStartSensor(MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "abs_obj_start_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        pos = self._sim.get_target_objs_start()
        return pos.reshape(-1)


@registry.register_sensor
class GoalSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


@registry.register_sensor
class AbsGoalSensor(MultiObjSensor):
    cls_uuid: str = "abs_obj_goal_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        _, pos = self._sim.get_targets()
        return pos.reshape(-1)


@registry.register_sensor
class JointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_joint_pos
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class HumanoidJointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "humanoid_joint_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.get_joint_transform()[0]
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class JointVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint_vel"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_velocity
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class EEPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_pos"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEPositionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        local_ee_pos = trans.inverted().transform_point(ee_pos)

        return np.array(local_ee_pos, dtype=np.float32)


@registry.register_sensor
class EETransformationSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_trans"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EETransformationSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,4),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
  
        # trans = self._sim.get_agent_data(
        #     self.agent_id
        # ).articulated_agent.base_transformation
        ee_trans = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
        )
        # local_ee_pos = trans.inverted().transform_point(ee_pos)

        return np.array(ee_trans, dtype=np.float32)
        





@registry.register_sensor
class ACTransformationSensor(UsesArticulatedAgentInterface, Sensor): #  articulated_camera transformation
    cls_uuid: str = "ac_trans"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ACTransformationSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,4),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
  
        # print("====== test observation ======", observations)

        cam_info = self._sim.get_agent_data(self.agent_id).articulated_agent.params.cameras["articulated_agent_arm_depth"]
        # print("============ cam info =============", cam_info)
        # Get the camera's attached link
        link_trans = self._sim.get_agent_data(self.agent_id).articulated_agent.sim_obj.get_link_scene_node(
            cam_info.attached_link_id
        ).transformation

        
        pos = cam_info.cam_offset_pos
        ori = cam_info.cam_orientation
        Mt = mn.Matrix4.translation(pos)
        Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
        My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
        Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
        offset_trans = Mt @ Mz @ My @ Mx

        coord_trans2 = mn.Matrix4.identity_init()

        coord_trans2[0,0] = 1
        coord_trans2[1,1] = -1
        coord_trans2[2,2] = -1

        cam_trans = link_trans @ offset_trans @ cam_info.relative_transform @ coord_trans2
        
        return np.asarray(cam_trans)




@registry.register_sensor
class GraspSensor(UsesArticulatedAgentInterface, Sensor): #  articulated_camera transformation
    cls_uuid: str = "grasp_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

        seed_feat_dim = 512
        checkpoint_path = "graspness_implementation/data/minkresunet_epoch10-1.tar"
        self.net = GraspNet(seed_feat_dim=seed_feat_dim, is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']        
        hfov = 65 * np.pi/180
        width = 320
        height = 240
        vfov = 2 * np.arctan(np.tan(hfov/2)*(width/height))
        fx = 1.0 / np.tan(hfov / 2.)
        fy = 1.0 / np.tan(vfov / 2.)

        self.camera = CameraInfo(320.0, 240.0, 320/2*fx, 240/2*fy, 320/2, 240/2, 1)
        self.net.eval()

        grasp_config = {"interval_size": 0.15, "observation_window_size": 200, "max_octree_threshold": 0.015, "min_octree_threshold": 0.01, "min_ori_threshold": np.pi/2  }

        self.fusion_grasp_gltree = GL_tree_grasp(grasp_config)
        self.time_step = 0

        self.current_goal_pos = None

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GraspSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box( 
            shape=(65,8), # 0-127 grasp info; 128 additional infomation
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_grasping_prediction(self, observation, target_point, camera_pose):
        rgb = observation["articulated_agent_arm_rgb"]
        depth = (observation["articulated_agent_arm_depth"][:,:,0])*10

        cloud = create_point_cloud_from_depth_image(depth, self.camera, organized=True)
        
        distance_threshold = 0.05

        # check the distance to the target point
        w_cloud = (cloud.reshape(-1, 3) @ camera_pose[:3,:3].T + camera_pose[:3,3] )
        distance_mask = (np.linalg.norm(w_cloud - target_point, axis=1) < distance_threshold).reshape(240,320)

        depth_mask = (depth > 0)

        valid_cloud = cloud[depth_mask]
        valid_rgb_cloud = rgb[depth_mask]        
        cloud_masked = depth_mask & distance_mask

        masked_cloud = cloud[cloud_masked]
        masked_rgb_cloud = rgb[cloud_masked]

        if np.sum(cloud_masked)<=450:
            return None, None, {"valid_cloud":valid_cloud.reshape(-1, 3), "valid_rgb_cloud":valid_rgb_cloud.reshape(-1, 3)}

        num_point = 1000
        voxel_size = 0.005
        voxel_size_cd = 0.005

        batch_size = 1
        collision_thresh = 0.01

        t_s = time.time()

        if np.sum(cloud_masked) >= num_point:
            idxs = np.random.choice(np.sum(cloud_masked), num_point, replace=False)
        else:
            idxs1 = np.arange(np.sum(cloud_masked))
            idxs2 = np.random.choice(np.sum(cloud_masked), num_point - np.sum(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = masked_cloud[idxs]
        rgb_sampled = masked_rgb_cloud[idxs]

        sample_data = {'point_clouds': cloud_sampled.astype(np.float32),
            'coors': cloud_sampled.astype(np.float32) / voxel_size,
            'feats': np.ones_like(cloud_sampled).astype(np.float32),
            }

        list_data = [sample_data]

        coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                    [d["feats"] for d in list_data])
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True)
        res = {
            "coors": coordinates_batch,
            "feats": features_batch,
            "quantize2original": quantize2original
        }


        def collate_fn_(batch):
            if type(batch[0]).__module__ == 'numpy':
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            elif isinstance(batch[0], container_abcs.Sequence):
                return [[torch.from_numpy(sample) for sample in b] for b in batch]
            elif isinstance(batch[0], container_abcs.Mapping):
                for key in batch[0]:
                    if key == 'coors' or key == 'feats':
                        continue
                    res[key] = collate_fn_([d[key] for d in batch])
                return res
        batch_data = collate_fn_(list_data)


        batch_interval = 100

        for key in batch_data:
            batch_data[key] = batch_data[key].to(self.device)

        with torch.no_grad():
            end_points = self.net(batch_data)
            
            if end_points is None:
                return None, None, {"valid_cloud":valid_cloud.reshape(-1, 3), "valid_rgb_cloud":valid_rgb_cloud.reshape(-1, 3)}

            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        for i in range(batch_size):
            # data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()

            gg = GraspGroup(preds)
            # collision detection
            if collision_thresh > 0:
                cloud = masked_cloud
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
                gg = gg[~collision_mask]

            gg = gg.sort_by_score()
            if gg.__len__() > 30:
                gg = gg[:30]

        if gg.__len__() == 0:
            return None, {"masked_cloud": masked_cloud.reshape(-1, 3), "masked_rgb_cloud": masked_rgb_cloud.reshape(-1, 3)}, {"valid_cloud":valid_cloud.reshape(-1, 3), "valid_rgb_cloud":valid_rgb_cloud.reshape(-1, 3)}


        return gg, {"masked_cloud": masked_cloud.reshape(-1, 3), "masked_rgb_cloud": masked_rgb_cloud.reshape(-1, 3)}, {"valid_cloud":valid_cloud.reshape(-1, 3), "valid_rgb_cloud":valid_rgb_cloud.reshape(-1, 3)}


    def get_observation(self, observations, episode, *args, **kwargs):
  
        # =============== goal pos ===============
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = np.asarray(global_T.inverted())
        goal_pos_abs = np.asarray(self._sim.get_target_objs_start()) 
    
        goal_pos = T_inv[:3, :3] @ goal_pos_abs[0, :] + T_inv[:3, 3]
        
        if self.current_goal_pos is None or np.any(self.current_goal_pos != goal_pos_abs):
            self.fusion_grasp_gltree.reset_gltree()
            self.current_goal_pos = goal_pos_abs
            self.time_step = 0

        cam_info = self._sim.get_agent_data(self.agent_id).articulated_agent.params.cameras["articulated_agent_arm_depth"]
        link_trans = self._sim.get_agent_data(self.agent_id).articulated_agent.sim_obj.get_link_scene_node(
            cam_info.attached_link_id
        ).transformation
        
        pos = cam_info.cam_offset_pos
        ori = cam_info.cam_orientation
        Mt = mn.Matrix4.translation(pos)
        Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
        My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
        Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
        offset_trans = Mt @ Mz @ My @ Mx

        coord_trans2 = mn.Matrix4.identity_init()

        coord_trans2[0,0] = 1
        coord_trans2[1,1] = -1
        coord_trans2[2,2] = -1

        ac_trans = np.asarray(link_trans @ offset_trans @ cam_info.relative_transform @ coord_trans2)

        # ================ get grasp pose =================
        grasp_num = 64
        grasp_indicate_array = np.zeros((grasp_num + 1, 3+4+1)) 

        gg = None

        if self.time_step % 10 ==0:
            gg, target_points, scene_points = self.get_grasping_prediction(observations, goal_pos_abs, ac_trans)

        if gg is not None:
            gg.transform(ac_trans)
            self.fusion_grasp_gltree.init_grasp_node(gg.grasp_group_array[:, 13:16])
            _ = self.fusion_grasp_gltree.add_grasps(gg.grasp_group_array, self.time_step)

            grasp_indicate_array[grasp_num, 0] = 0
            grasp_indicate_array[grasp_num, 1] = 0

        if self.fusion_grasp_gltree.get_node_numbder()>0:

            grasp_indicate_array[grasp_num, 4] = 1

            fused_grasp_group_array =  self.fusion_grasp_gltree.all_grasps_array()
            temp_grasp_indicate_array = np.zeros((fused_grasp_group_array.shape[0], 3+4+1))

            temp_grasp_indicate_array[:, :3] = fused_grasp_group_array[:, :3] @ T_inv[:3, :3].T + T_inv[:3, 3]
            temp_grasp_indicate_array[:, 7] = fused_grasp_group_array[:, 12]


            for i in range(fused_grasp_group_array.shape[0]):
                temp_grasp_indicate_array[i, 3:7] = Rotation.from_matrix(T_inv[:3,:3].T @ fused_grasp_group_array[i,3:3+9].reshape(3,3)).as_quat()

            num_rows = temp_grasp_indicate_array.shape[0] 
            if num_rows < grasp_num:
                sample_indices = np.random.choice(num_rows, size=grasp_num, replace=True)  
            else:
                sample_indices = np.random.choice(num_rows, size=grasp_num, replace=False) 

            grasp_indicate_array[:grasp_num, :] = temp_grasp_indicate_array[sample_indices, :] 
        
        else:
            grasp_indicate_array[:grasp_num, :3] = goal_pos
            grasp_indicate_array[:grasp_num, 3:7] = Rotation.from_matrix(np.eye(3)).as_quat()
            grasp_indicate_array[:grasp_num, 7] = 0.05 # default is zero

        grasp_indicate_array[grasp_num, 2] = self.fusion_grasp_gltree.get_node_numbder()
        grasp_indicate_array[grasp_num, 3] = self.fusion_grasp_gltree.get_edge_number()

        grasp_indicate_array[grasp_num, 4] = self.fusion_grasp_gltree.get_fused_count()
        grasp_indicate_array[grasp_num, 5] = self.fusion_grasp_gltree.get_totla_score()

        self.time_step += 1

        return grasp_indicate_array


@registry.register_sensor
class RelativeRestingPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "relative_resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RelativeRestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        base_trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = ( 
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        local_ee_pos = base_trans.inverted().transform_point(ee_pos)

        relative_desired_resting = task.desired_resting - local_ee_pos

        return np.array(relative_desired_resting, dtype=np.float32)


@registry.register_sensor
class RestingPositionSensor(Sensor):
    """
    Desired resting position in the articulated_agent coordinate frame.
    """

    cls_uuid: str = "resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        return np.array(task.desired_resting, dtype=np.float32)


@registry.register_sensor
class LocalizationSensor(UsesArticulatedAgentInterface, Sensor):
    """
    The position and angle of the articulated_agent in world coordinates.
    """

    cls_uuid = "localization_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return LocalizationSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        articulated_agent = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent
        T = articulated_agent.base_transformation
        forward = np.array([1.0, 0, 0])
        heading_angle = get_angle_to_pos(T.transform_vector(forward))
        return np.array(
            [*articulated_agent.base_pos, heading_angle], dtype=np.float32
        )


@registry.register_sensor
class IsHoldingSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Binary if the robot is holding an object or grasped onto an articulated object.
    """

    cls_uuid: str = "is_holding"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return IsHoldingSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        return np.array(
            int(self._sim.get_agent_data(self.agent_id).grasp_mgr.is_grasped),
            dtype=np.float32,
        ).reshape((1,))


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """
    Euclidean distance from the target object to the goal.
    """

    cls_uuid: str = "object_to_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        distances = np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        self._metric = {str(idx): dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class GfxReplayMeasure(Measure):
    cls_uuid: str = "gfx_replay_keyframes_string"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._enable_gfx_replay_save = (
            self._sim.sim_config.sim_cfg.enable_gfx_replay_save
        )
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GfxReplayMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._gfx_replay_keyframes_string = None
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        if not task._is_episode_active and self._enable_gfx_replay_save:
            self._metric = (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        else:
            self._metric = ""

    def get_metric(self, force_get=False):
        if force_get and self._enable_gfx_replay_save:
            return (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        return super().get_metric()


@registry.register_measure
class ObjAtGoal(Measure):
    """
    Returns if the target object is at the goal (binary) for each of the target
    objects in the scene.
    """

    cls_uuid: str = "obj_at_goal"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._succ_thresh = self._config.succ_thresh
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAtGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ObjectToGoalDistance.cls_uuid,
            ],
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        obj_to_goal_dists = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()

        self._metric = {
            str(idx): dist < self._succ_thresh
            for idx, dist in obj_to_goal_dists.items()
        }


@registry.register_measure
class EndEffectorToGoalDistance(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "ee_to_goal_distance"

    def __init__(self, sim, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, observations, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        idxs, goals = self._sim.get_targets()

        distances = np.linalg.norm(goals - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class EndEffectorToObjectDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the end-effector and all current target object COMs.
    """

    cls_uuid: str = "ee_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        distances = np.linalg.norm(target_pos - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class EndEffectorToTargetEndEffectorDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the current end-effector position 
    and the target end effector position.
    """

    cls_uuid: str = "ee_to_target_ee_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToTargetEndEffectorDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        current_ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        ) 

        distances = np.linalg.norm(target_ee_pos - current_ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in zip(idxs, distances)}

@registry.register_measure
class EndEffectorToRestDistance(Measure):
    """
    Distance between current end effector position and position where end effector rests within the robot body.
    """

    cls_uuid: str = "ee_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        self._metric = rest_dist


@registry.register_measure
class ReturnToRestDistance(UsesArticulatedAgentInterface, Measure):
    """
    Distance between end-effector and resting position if the articulated agent is holding the object.
    """

    cls_uuid: str = "return_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ReturnToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        snapped_id = self._sim.get_agent_data(self.agent_id).grasp_mgr.snap_idx
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
        picked_correct = snapped_id == abs_targ_obj_idx

        if picked_correct:
            self._metric = rest_dist
        else:
            T_inv = (
                self._sim.get_agent_data(self.agent_id)
                .articulated_agent.ee_transform()
                .inverted()
            )
            idxs, _ = self._sim.get_targets()
            scene_pos = self._sim.get_scene_pos()
            pos = scene_pos[idxs][0]
            pos = T_inv.transform_point(pos)

            self._metric = np.linalg.norm(task.desired_resting - pos)


@registry.register_measure
class RobotCollisions(UsesArticulatedAgentInterface, Measure):
    """
    Returns a dictionary with the counts for different types of collisions.
    """

    cls_uuid: str = "robot_collisions"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotCollisions.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_coll_info = CollisionDetails()
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        cur_coll_info = self._task.get_cur_collision_info(self.agent_id)
        self._accum_coll_info += cur_coll_info
        self._metric = {
            "total_collisions": self._accum_coll_info.total_collisions,
            "robot_obj_colls": self._accum_coll_info.robot_obj_colls,
            "robot_scene_colls": self._accum_coll_info.robot_scene_colls,
            "obj_scene_colls": self._accum_coll_info.obj_scene_colls,
        }


@registry.register_measure
class RobotForce(UsesArticulatedAgentInterface, Measure):
    """
    The amount of force in newton's accumulatively applied by the robot.
    """

    cls_uuid: str = "articulated_agent_force"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._count_obj_collisions = self._task._config.count_obj_collisions
        self._min_force = self._config.min_force
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotForce.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_force = 0.0
        self._prev_force = None
        self._cur_force = None
        self._add_force = None
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    @property
    def add_force(self):
        return self._add_force

    def update_metric(self, *args, episode, task, observations, **kwargs):
        articulated_agent_force, _, overall_force = self._task.get_coll_forces(
            self.agent_id
        )
        if self._count_obj_collisions:
            self._cur_force = overall_force
        else:
            self._cur_force = articulated_agent_force

        if self._prev_force is not None:
            self._add_force = self._cur_force - self._prev_force
            if self._add_force > self._min_force:
                self._accum_force += self._add_force
                self._prev_force = self._cur_force
            elif self._add_force < 0.0:
                self._prev_force = self._cur_force
            else:
                self._add_force = 0.0
        else:
            self._prev_force = self._cur_force
            self._add_force = 0.0

        self._metric = {
            "accum": self._accum_force,
            "instant": self._cur_force,
        }


@registry.register_measure
class NumStepsMeasure(Measure):
    """
    The number of steps elapsed in the current episode.
    """

    cls_uuid: str = "num_steps"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumStepsMeasure.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric += 1


@registry.register_measure
class ForceTerminate(Measure):
    """
    If the accumulated force throughout this episode exceeds the limit.
    """

    cls_uuid: str = "force_terminate"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._max_accum_force = self._config.max_accum_force
        self._max_instant_force = self._config.max_instant_force
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ForceTerminate.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RobotForce.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        force_info = task.measurements.measures[
            RobotForce.cls_uuid
        ].get_metric()
        accum_force = force_info["accum"]
        instant_force = force_info["instant"]
        if self._max_accum_force > 0 and accum_force > self._max_accum_force:
            rearrange_logger.debug(
                f"Force threshold={self._max_accum_force} exceeded with {accum_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        elif (
            self._max_instant_force > 0
            and instant_force > self._max_instant_force
        ):
            rearrange_logger.debug(
                f"Force instant threshold={self._max_instant_force} exceeded with {instant_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        else:
            self._metric = False


@registry.register_measure
class DidViolateHoldConstraintMeasure(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "did_violate_hold_constraint"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidViolateHoldConstraintMeasure.cls_uuid

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim

        super().__init__(*args, sim=sim, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, **kwargs):
        self._metric = self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint()


class RearrangeReward(UsesArticulatedAgentInterface, Measure):
    """
    An abstract class defining some measures that are always a part of any
    reward function in the Habitat 2.0 tasks.
    """

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._force_pen = self._config.force_pen
        self._max_force_pen = self._config.max_force_pen
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = 0.0

        reward += self._get_coll_reward()

        if self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint():
            reward -= self._config.constraint_violate_pen

        force_terminate = task.measurements.measures[
            ForceTerminate.cls_uuid
        ].get_metric()
        if force_terminate:
            reward -= self._config.force_end_pen

        self._metric = reward

    def _get_coll_reward(self):
        reward = 0

        force_metric = self._task.measurements.measures[RobotForce.cls_uuid]
        # Penalize the force that was added to the accumulated force at the
        # last time step.
        reward -= max(
            0,  # This penalty is always positive
            min(
                self._force_pen * force_metric.add_force,
                self._max_force_pen,
            ),
        )
        return reward


@registry.register_measure
class DoesWantTerminate(Measure):
    """
    Returns 1 if the agent has called the stop action and 0 otherwise.
    """

    cls_uuid: str = "does_want_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DoesWantTerminate.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.actions["rearrange_stop"].does_want_terminate


@registry.register_measure
class BadCalledTerminate(Measure):
    """
    Returns 0 if the agent has called the stop action when the success
    condition is also met or not called the stop action when the success
    condition is not met. Returns 1 otherwise.
    """

    cls_uuid: str = "bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return BadCalledTerminate.cls_uuid

    def __init__(self, config, task, *args, **kwargs):
        super().__init__(**kwargs)
        self._success_measure_name = task._config.success_measure
        self._config = config

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid, self._success_measure_name],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        is_succ = task.measurements.measures[
            self._success_measure_name
        ].get_metric()

        self._metric = (not is_succ) and does_action_want_stop
