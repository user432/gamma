#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToObjectDistance,
    EndEffectorToRestDistance,
    ForceTerminate,
    RearrangeReward,
    RobotForce
    )
from habitat.tasks.rearrange.utils import rearrange_logger

import numpy as np
import cv2

@registry.register_measure
class DidPickObjectMeasure(Measure):
    cls_uuid: str = "did_pick_object"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidPickObjectMeasure.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self._did_pick = False
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        self._did_pick = self._did_pick or self._sim.grasp_mgr.is_grasped
        self._metric = int(self._did_pick)


@registry.register_measure
class GraspingDistanceReward(RearrangeReward):
    cls_uuid: str = "grasp_distance_measure"

    def __init__(self, *args, sim, config, task, **kwargs):

        self._metric = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GraspingDistanceReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):

        if "grasp_sensor" in observations: # No success grasping is predicted.
            # move the arm to the target 
            grasp_group_array = observations["grasp_sensor"][:64,:]
            grasp_distance = np.linalg.norm(grasp_group_array[:,:3], axis=1)
            self._metric = np.min(grasp_distance)
        else:
            self._metric = 0


@registry.register_measure
class GraspingOrientationReward(RearrangeReward):
    cls_uuid: str = "grasp_ori_measure"

    def __init__(self, *args, sim, config, task, **kwargs):

        self._metric = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GraspingOrientationReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):

        if "grasp_sensor" in observations: # No success grasping is predicted.
            # move the arm to the target 
            grasp_group_array = observations["grasp_sensor"][:64,:]
            grasp_orientation = np.linalg.norm(grasp_group_array[:,3:6], axis=1)
            self._metric = np.min(grasp_orientation)
        else:
            self._metric = 0


@registry.register_measure
class GraspingNodesReward(RearrangeReward):
    cls_uuid: str = "grasp_nodes_measure"

    def __init__(self, *args, sim, config, task, **kwargs):

        self._metric = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GraspingNodesReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):

        if "grasp_sensor" in observations: # No success grasping is predicted.
            self._metric = observations["grasp_sensor"][64, 2]
        else:
            self._metric = 0


@registry.register_measure
class GraspingEdgesReward(RearrangeReward):
    cls_uuid: str = "grasp_edges_measure"

    def __init__(self, *args, sim, config, task, **kwargs):

        self._metric = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GraspingEdgesReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):

        if "grasp_sensor" in observations: # No success grasping is predicted.
            self._metric = observations["grasp_sensor"][64, 3]
        else:
            self._metric = 0



@registry.register_measure
class GraspingFusedCountReward(RearrangeReward):
    cls_uuid: str = "grasp_fused_count_measure"

    def __init__(self, *args, sim, config, task, **kwargs):

        self._metric = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GraspingFusedCountReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):

        if "grasp_sensor" in observations: # No success grasping is predicted.
            self._metric = observations["grasp_sensor"][64, 4]
        else:
            self._metric = 0


@registry.register_measure
class GraspingTotalScoreReward(RearrangeReward):
    cls_uuid: str = "grasp_total_score_measure"

    def __init__(self, *args, sim, config, task, **kwargs):

        self._metric = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GraspingTotalScoreReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):

        if "grasp_sensor" in observations: # No success grasping is predicted.
            self._metric = observations["grasp_sensor"][64, 5]
        else:
            self._metric = 0



@registry.register_measure
class RearrangePickReward(RearrangeReward):
    cls_uuid: str = "pick_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self.cur_dist = -1.0
        self._prev_picked = False
        self._metric = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

        self.prev_good_grasping_distance_measure = None
        self.prev_good_grasping_info_gain_measure = None

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToObjectDistance.cls_uuid,
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )

        self.prev_good_grasping_distance_measure = None
        self.prev_good_grasping_info_gain_measure = None
        self.prev_good_grasping_orientation_measure = None

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

        grasp_distance_measure = task.measurements.measures[GraspingDistanceReward.cls_uuid].get_metric()

        grasping_reward = 0
        timestep = 0
        grasping_num = 64

        # add grasping reward
        if "grasp_sensor" in observations:

            # move the arm to the target 
            grasp_group_array = observations["grasp_sensor"][:grasping_num,:]
            grasp_distance = np.linalg.norm(grasp_group_array[:,:3], axis=1)
            grasp_orientation = np.linalg.norm(grasp_group_array[:,3:6], axis=1)

            # distance and orientation reward
            grasp_reward_distance = np.min(grasp_distance * (1 - grasp_group_array[:, -1]))
            grasp_reward_orientation = np.min(grasp_orientation * (1 - grasp_group_array[:, -1]))
            
            # infogain reward
            grasp_reward_infoGain = (observations["grasp_sensor"][grasping_num, 2] + observations["grasp_sensor"][grasping_num, 3]\
                                        + observations["grasp_sensor"][grasping_num, 4] + observations["grasp_sensor"][grasping_num, 5])/60
            
            weight = 1 / (1 + np.exp(0.5 - timestep / 300))

            if self.prev_good_grasping_distance_measure is not None \
                and self.prev_good_grasping_info_gain_measure is not None\
                 and self.prev_good_grasping_orientation_measure is not None:

                grasp_distance_difference = (self.prev_good_grasping_distance_measure - grasp_reward_distance)
                grasp_orientation_difference = (self.prev_good_grasping_orientation_measure - grasp_reward_orientation)
                grasp_infoGain_difference = (grasp_reward_infoGain - self.prev_good_grasping_info_gain_measure)
                
                grasping_reward = 100 *((grasp_distance_difference + grasp_orientation_difference)*(1 - weight) + (grasp_infoGain_difference * weight))
                
            self.prev_good_grasping_distance_measure = grasp_reward_distance 
            self.prev_good_grasping_info_gain_measure = grasp_reward_infoGain
            self.prev_good_grasping_orientation_measure = grasp_reward_orientation

            
            self._metric += grasping_reward
            timestep += 1
            if grasp_distance_measure < 0.1:
                self._task.should_end = True

        elif observations["grasp_sensor"][grasping_num, 4] == 0:
            # add ee-to-goal distance reward
            
            ee_to_object_distance = task.measurements.measures[
                EndEffectorToObjectDistance.cls_uuid
            ].get_metric()

            dist_to_goal = ee_to_object_distance[str(task.abs_targ_idx)]

            if self._config.use_diff:
                if self.cur_dist < 0:
                    dist_diff = 0.0
                else:
                    dist_diff = self.cur_dist - dist_to_goal

                # Filter out the small fluctuations
                dist_diff = round(dist_diff, 3)
                self._metric += self._config.dist_reward * dist_diff
            else:
                self._metric -= self._config.dist_reward * dist_to_goal
            self.cur_dist = dist_to_goal


@registry.register_measure
class RearrangePickSuccess(Measure):
    cls_uuid: str = "pick_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._prev_ee_pos = None
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [EndEffectorToObjectDistance.cls_uuid]
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        grasp_distance = task.measurements.measures[
            GraspingDistanceReward.cls_uuid
        ].get_metric()

        ee_to_object_distance = task.measurements.measures[
                EndEffectorToObjectDistance.cls_uuid
            ].get_metric()

        if grasp_distance == None:
            self._metric = False
        else:
            self._metric = grasp_distance < 0.11
