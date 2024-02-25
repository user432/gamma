#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.articulated_agent_action import (
    ArticulatedAgentAction,
)

# flake8: noqa
# These actions need to be imported since there is a Python evaluation
# statement which dynamically creates the desired grip controller.
from habitat.tasks.rearrange.actions.grip_actions import (
    GazeGraspAction,
    GripSimulatorTaskAction,
    MagicGraspAction,
    SuctionGraspAction,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import rearrange_collision, rearrange_logger


@registry.register_task_action
class EmptyAction(ArticulatedAgentAction):
    """A No-op action useful for testing and in some controllers where we want
    to wait before the next operation.
    """

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "empty_action": spaces.Box(
                    shape=(1,),
                    low=-1,
                    high=1,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.empty)


@registry.register_task_action
class RearrangeStopAction(SimulatorTaskAction):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    def step(self, task, *args, is_last_action, **kwargs):
        should_stop = kwargs.get("rearrange_stop", [1.0])
        if should_stop[0] > 0.0:
            rearrange_logger.debug(
                "Rearrange stop action requesting episode stop."
            )
            self.does_want_terminate = True

        if is_last_action:
            return self._sim.step(HabitatSimActions.rearrange_stop)
        else:
            return {}


@registry.register_task_action
class ArmAction(ArticulatedAgentAction):
    """An arm control and grip control into one action space."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        arm_controller_cls = eval(self._config.arm_controller)
        self._sim: RearrangeSim = sim
        self.arm_ctrlr = arm_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )

        if self._config.grip_controller is not None:
            grip_controller_cls = eval(self._config.grip_controller)
            self.grip_ctrlr: Optional[
                GripSimulatorTaskAction
            ] = grip_controller_cls(*args, config=config, sim=sim, **kwargs)
        else:
            self.grip_ctrlr = None

        self.disable_grip = False
        if "disable_grip" in config:
            self.disable_grip = config["disable_grip"]

    def reset(self, *args, **kwargs):
        self.arm_ctrlr.reset(*args, **kwargs)
        if self.grip_ctrlr is not None:
            self.grip_ctrlr.reset(*args, **kwargs)

    @property
    def action_space(self):
        action_spaces = {
            self._action_arg_prefix
            + "arm_action": self.arm_ctrlr.action_space,
        }
        if self.grip_ctrlr is not None and self.grip_ctrlr.requires_action:
            action_spaces[
                self._action_arg_prefix + "grip_action"
            ] = self.grip_ctrlr.action_space
        return spaces.Dict(action_spaces)

    def step(self, is_last_action, *args, **kwargs):
        arm_action = kwargs[self._action_arg_prefix + "arm_action"]
        self.arm_ctrlr.step(arm_action)
        if self.grip_ctrlr is not None and not self.disable_grip:
            grip_action = kwargs[self._action_arg_prefix + "grip_action"]
            self.grip_ctrlr.step(grip_action)
        if is_last_action:
            return self._sim.step(HabitatSimActions.arm_action)
        else:
            return {}


@registry.register_task_action
class ArmRelPosAction(ArticulatedAgentAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._delta_pos_limit = self._config.delta_pos_limit

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._delta_pos_limit

        # The actual joint positions
        self._sim: RearrangeSim
        self.cur_articulated_agent.arm_motor_pos = (
            delta_pos + self.cur_articulated_agent.arm_motor_pos
        )


@registry.register_task_action
class ArmRelPosMaskAction(ArticulatedAgentAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._delta_pos_limit = self._config.delta_pos_limit
        self._arm_joint_mask = self._config.arm_joint_mask

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._delta_pos_limit

        mask_delta_pos = np.zeros(len(self._arm_joint_mask))
        src_idx = 0
        tgt_idx = 0
        for mask in self._arm_joint_mask:
            if mask == 0:
                tgt_idx += 1
                src_idx += 1
                continue
            mask_delta_pos[tgt_idx] = delta_pos[src_idx]
            tgt_idx += 1
            src_idx += 1

        # Although habitat_sim will prevent the motor from exceeding limits,
        # clip the motor joints first here to prevent the arm from being unstable.
        min_limit, max_limit = self.cur_articulated_agent.arm_joint_limits
        target_arm_pos = (
            mask_delta_pos + self.cur_articulated_agent.arm_motor_pos
        )
        set_arm_pos = np.clip(target_arm_pos, min_limit, max_limit)

        # The actual joint positions
        self._sim: RearrangeSim
        self.cur_articulated_agent.arm_motor_pos = set_arm_pos


@registry.register_task_action
class ArmRelPosKinematicAction(ArticulatedAgentAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._delta_pos_limit = self._config.delta_pos_limit
        self._should_clip = self._config.get("should_clip", True)

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._should_clip:
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._delta_pos_limit
        self._sim: RearrangeSim

        set_arm_pos = delta_pos + self.cur_articulated_agent.arm_joint_pos
        self.cur_articulated_agent.arm_joint_pos = set_arm_pos
        self.cur_articulated_agent.fix_joint_values = set_arm_pos


@registry.register_task_action
class ArmAbsPosAction(ArticulatedAgentAction):
    """
    The arm motor targets are directly set to the joint configuration specified
    by the action.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, set_pos, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self.cur_articulated_agent.arm_motor_pos = set_pos


@registry.register_task_action
class ArmAbsPosKinematicAction(ArticulatedAgentAction):
    """
    The arm is kinematically directly set to the joint configuration specified
    by the action.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, set_pos, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self.cur_articulated_agent.arm_joint_pos = set_pos


@registry.register_task_action
class ArmRelPosKinematicReducedActionStretch(ArticulatedAgentAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action and the mask. This function is used for Stretch.
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.last_arm_action = None
        self._delta_pos_limit = self._config.delta_pos_limit
        self._should_clip = self._config.get("should_clip", True)
        self._arm_joint_mask = self._config.arm_joint_mask

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.last_arm_action = None

    @property
    def action_space(self):
        self.step_c = 0
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._should_clip:
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._delta_pos_limit
        self._sim: RearrangeSim

        # Expand delta_pos based on mask
        expanded_delta_pos = np.zeros(len(self._arm_joint_mask))
        src_idx = 0
        tgt_idx = 0
        for mask in self._arm_joint_mask:
            if mask == 0:
                tgt_idx += 1
                src_idx += 1
                continue
            expanded_delta_pos[tgt_idx] = delta_pos[src_idx]
            tgt_idx += 1
            src_idx += 1

        min_limit, max_limit = self.cur_articulated_agent.arm_joint_limits
        set_arm_pos = (
            expanded_delta_pos + self.cur_articulated_agent.arm_motor_pos
        )
        # Perform roll over to the joints so that the user cannot control
        # the motor 2, 3, 4 for the arm.
        if expanded_delta_pos[0] >= 0:
            for i in range(3):
                if set_arm_pos[i] > max_limit[i]:
                    set_arm_pos[i + 1] += set_arm_pos[i] - max_limit[i]
                    set_arm_pos[i] = max_limit[i]
        else:
            for i in range(3):
                if set_arm_pos[i] < min_limit[i]:
                    set_arm_pos[i + 1] -= min_limit[i] - set_arm_pos[i]
                    set_arm_pos[i] = min_limit[i]
        set_arm_pos = np.clip(set_arm_pos, min_limit, max_limit)

        self.cur_articulated_agent.arm_motor_pos = set_arm_pos


@registry.register_task_action
class BaseVelAction(ArticulatedAgentAction):
    """
    The articulated agent base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True
        self._allow_dyn_slide = self._config.get("allow_dyn_slide", True)
        self._lin_speed = self._config.lin_speed
        self._ang_speed = self._config.ang_speed
        self._allow_back = self._config.allow_back

    @property
    def action_space(self):
        lim = 20
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "base_vel": spaces.Box(
                    shape=(2,), low=-lim, high=lim, dtype=np.float32
                )
            }
        )

    def _capture_articulated_agent_state(self):
        return {
            "forces": self.cur_articulated_agent.sim_obj.joint_forces,
            "vel": self.cur_articulated_agent.sim_obj.joint_velocities,
            "pos": self.cur_articulated_agent.sim_obj.joint_positions,
        }

    def _set_articulated_agent_state(self, set_dat):
        self.cur_articulated_agent.sim_obj.joint_positions = set_dat["forces"]
        self.cur_articulated_agent.sim_obj.joint_velocities = set_dat["vel"]
        self.cur_articulated_agent.sim_obj.joint_forces = set_dat["pos"]

    def update_base(self):
        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_articulated_agent_state()

        trans = self.cur_articulated_agent.sim_obj.transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )

        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )
        end_pos = self._sim.step_filter(
            rigid_state.translation, target_rigid_state.translation
        )

        # Offset the base
        end_pos -= self.cur_articulated_agent.params.base_offset

        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(), end_pos
        )
        self.cur_articulated_agent.sim_obj.transformation = target_trans

        if not self._allow_dyn_slide:
            # Check if in the new articulated_agent state the arm collides with anything.
            # If so we have to revert back to the previous transform
            self._sim.internal_step(-1)
            colls = self._sim.get_collisions()
            did_coll, _ = rearrange_collision(
                colls, self._sim.snapped_obj_id, False
            )
            if did_coll:
                # Don't allow the step, revert back.
                self._set_articulated_agent_state(before_trans_state)
                self.cur_articulated_agent.sim_obj.transformation = trans
        if self.cur_grasp_mgr.snap_idx is not None:
            # Holding onto an object, also kinematically update the object.
            # object.
            self.cur_grasp_mgr.update_object_to_grasp()

        if self.cur_articulated_agent._base_type == "leg":
            # Fix the leg joints
            self.cur_articulated_agent.leg_joint_pos = (
                self.cur_articulated_agent.params.leg_init_params
            )

    def step(self, *args, is_last_action, **kwargs):
        lin_vel, ang_vel = kwargs[self._action_arg_prefix + "base_vel"]
        lin_vel = np.clip(lin_vel, -1, 1) * self._lin_speed
        ang_vel = np.clip(ang_vel, -1, 1) * self._ang_speed
        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()

        if is_last_action:
            return self._sim.step(HabitatSimActions.base_velocity)
        else:
            return {}


@registry.register_task_action
class BaseVelNonCylinderAction(ArticulatedAgentAction):
    """
    The articulated agent base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True
        self._allow_dyn_slide = self._config.get("allow_dyn_slide", True)
        self._allow_back = self._config.allow_back
        self._collision_threshold = self._config.collision_threshold
        self._longitudinal_lin_speed = self._config.longitudinal_lin_speed
        self._lateral_lin_speed = self._config.lateral_lin_speed
        self._ang_speed = self._config.ang_speed
        self._navmesh_offset = self._config.navmesh_offset
        self._enable_lateral_move = self._config.enable_lateral_move

    @property
    def action_space(self):
        lim = 20
        if self._enable_lateral_move:
            return spaces.Dict(
                {
                    self._action_arg_prefix
                    + "base_vel": spaces.Box(
                        shape=(3,), low=-lim, high=lim, dtype=np.float32
                    )
                }
            )
        else:
            return spaces.Dict(
                {
                    self._action_arg_prefix
                    + "base_vel": spaces.Box(
                        shape=(2,), low=-lim, high=lim, dtype=np.float32
                    )
                }
            )

    def collision_check(
        self, trans, target_trans, target_rigid_state, compute_sliding
    ):
        """
        trans: the transformation of the current location of the robot
        target_trans: the transformation of the target location of the robot given the center original Navmesh
        target_rigid_state: the target state of the robot given the center original Navmesh
        compute_sliding: if we want to compute sliding or not
        """
        # Get the offset positions
        num_check_cylinder = len(self._navmesh_offset)
        nav_pos_3d = [
            np.array([xz[0], 0.0, xz[1]]) for xz in self._navmesh_offset
        ]
        cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
        goal_pos = [target_trans.transform_point(xyz) for xyz in nav_pos_3d]

        # For step filter of offset positions
        end_pos = []
        for i in range(num_check_cylinder):
            pos = self._sim.step_filter(cur_pos[i], goal_pos[i])
            # Sanitize the height
            pos[1] = 0.0
            cur_pos[i][1] = 0.0
            goal_pos[i][1] = 0.0
            end_pos.append(pos)

        # Planar move distance clamped by NavMesh
        move = []
        for i in range(num_check_cylinder):
            move.append((end_pos[i] - goal_pos[i]).length())

        # For detection of linear or angualr velocities
        # There is a collision if the difference between the clamped NavMesh position and target position is too great for any point.
        diff = len([v for v in move if v > self._collision_threshold])

        if diff > 0:
            # Wrap the move direction if we use sliding
            # Find the largest diff moving direction, which means that there is a collision in that cylinder
            if compute_sliding:
                max_idx = np.argmax(move)
                move_vec = end_pos[max_idx] - cur_pos[max_idx]
                new_end_pos = trans.translation + move_vec
                return True, mn.Matrix4.from_(
                    target_rigid_state.rotation.to_matrix(), new_end_pos
                )
            return True, trans
        else:
            return False, target_trans

    def update_base(self, if_rotation):
        """
        Update the base of the robot
        if_rotation: if the robot is rotating or not
        """
        # Get the control frequency
        ctrl_freq = self._sim.ctrl_freq
        # Get the current transformation
        trans = self.cur_articulated_agent.sim_obj.transformation
        # Get the current rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        # Integrate to get target rigid state
        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )
        # Get the traget transformation based on the target rigid state
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            target_rigid_state.translation,
        )
        # We do sliding only if we allow the robot to do sliding and current
        # robot is not rotating
        compute_sliding = self._allow_dyn_slide and not if_rotation
        # Check if there is a collision
        did_coll, new_target_trans = self.collision_check(
            trans, target_trans, target_rigid_state, compute_sliding
        )
        # Update the base
        self.cur_articulated_agent.sim_obj.transformation = new_target_trans

        if self.cur_grasp_mgr.snap_idx is not None:
            # Holding onto an object, also kinematically update the object.
            # object.
            self.cur_grasp_mgr.update_object_to_grasp()

        if self.cur_articulated_agent._base_type == "leg":
            # Fix the leg joints
            self.cur_articulated_agent.leg_joint_pos = (
                self.cur_articulated_agent.params.leg_init_params
            )

    def step(self, *args, is_last_action, **kwargs):
        lateral_lin_vel = 0.0
        if self._enable_lateral_move:
            longitudinal_lin_vel, lateral_lin_vel, ang_vel = kwargs[
                self._action_arg_prefix + "base_vel"
            ]
        else:
            longitudinal_lin_vel, ang_vel = kwargs[
                self._action_arg_prefix + "base_vel"
            ]

        longitudinal_lin_vel = (
            np.clip(longitudinal_lin_vel, -1, 1) * self._longitudinal_lin_speed
        )
        lateral_lin_vel = (
            np.clip(lateral_lin_vel, -1, 1) * self._lateral_lin_speed
        )
        #lateral_lin_vel = 0.0
        ang_vel = np.clip(ang_vel, -1, 1) * self._ang_speed
        if not self._allow_back:
            longitudinal_lin_vel = np.maximum(longitudinal_lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(
            longitudinal_lin_vel, 0, -lateral_lin_vel
        )
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if (
            longitudinal_lin_vel != 0.0
            or lateral_lin_vel != 0.0
            or ang_vel != 0.0
        ):
            self.update_base(ang_vel != 0.0)

        if is_last_action:
            return self._sim.step(HabitatSimActions.base_velocity)
        else:
            return {}


@registry.register_task_action
class ArmEEAction(ArticulatedAgentAction):
    """Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm."""

    def __init__(self, *args, sim: RearrangeSim, **kwargs):
        self.ee_target: Optional[np.ndarray] = None
        self.ee_index: Optional[int] = 0
        super().__init__(*args, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self._render_ee_target = self._config.get("render_ee_target", False)
        self._ee_ctrl_lim = self._config.ee_ctrl_lim

    def reset(self, *args, **kwargs):
        super().reset()
        cur_ee = self._ik_helper.calc_fk(
            np.array(self._sim.articulated_agent.arm_joint_pos)
        )

        self.ee_target = cur_ee

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-1, high=1, dtype=np.float32)

    def apply_ee_constraints(self):
        self.ee_target = np.clip(
            self.ee_target,
            self._sim.articulated_agent.params.ee_constraint[
                self.ee_index, :, 0
            ],
            self._sim.articulated_agent.params.ee_constraint[
                self.ee_index, :, 1
            ],
        )

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> None:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        joint_pos = np.array(self._sim.articulated_agent.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        self._sim.articulated_agent.arm_motor_pos = des_joint_pos

    def step(self, ee_pos, **kwargs):
        ee_pos = np.clip(ee_pos, -1, 1)
        ee_pos *= self._ee_ctrl_lim
        self.set_desired_ee_pos(ee_pos)

        if self._render_ee_target:
            global_pos = self._sim.articulated_agent.base_transformation.transform_point(
                self.ee_target
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )


@registry.register_task_action
class HumanoidJointAction(ArticulatedAgentAction):
    def __init__(self, *args, sim: RearrangeSim, **kwargs):
        super().__init__(*args, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.num_joints = self._config.num_joints

    def reset(self, *args, **kwargs):
        super().reset()

    @property
    def action_space(self):
        num_joints = self.num_joints
        num_dim_transform = 16
        # The action space is the number of joints plus 16 for a 4x4 transformtion matrix for the base
        return spaces.Dict(
            {
                "human_joints_trans": spaces.Box(
                    shape=(4 * num_joints + num_dim_transform,),
                    low=-1,
                    high=1,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, is_last_action, **kwargs):
        r"""
        Updates the joint rotations and root transformation of the humanoid.
        :param self._action_arg_prefix+human_joints_trans: Array of size
            (num_joints*4)+32. The last 32 dimensions define two 4x4 root
            transformation matrices, a base transform that controls the base
            of the character, and an offset transform, that controls
            a transformation offset that comes from the MOCAP pose.
            The first elements correspond to a flattened list of quaternions for each joint.
            When the array is all 0 it keeps the previous joint rotation and transform.
        :param is_last_action: whether this is the last action before calling environment
          step
        """
        human_joints_trans = kwargs[
            self._action_arg_prefix + "human_joints_trans"
        ]
        new_joints = human_joints_trans[:-16]
        new_pos_transform_base = human_joints_trans[-16:]
        new_pos_transform_offset = human_joints_trans[-32:-16]

        # When the array is all 0, this indicates we are not setting
        # the human joint
        if np.array(new_pos_transform_offset).sum() != 0:
            vecs_base = [
                mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            vecs_offset = [
                mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            new_transform_offset = mn.Matrix4(*vecs_offset)
            new_transform_base = mn.Matrix4(*vecs_base)
            if (
                new_transform_offset.is_rigid_transformation()
                and new_transform_base.is_rigid_transformation()
            ):
                # TODO: this will cause many sampled actions to be invalid
                # Maybe we should update the sampling mechanism
                self.cur_articulated_agent.set_joint_transform(
                    new_joints, new_transform_offset, new_transform_base
                )

        if is_last_action:
            return self._sim.step(HabitatSimActions.humanoidjoint_action)
        else:
            return {}
