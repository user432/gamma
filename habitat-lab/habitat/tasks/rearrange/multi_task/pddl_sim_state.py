#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, cast

import magnum as mn
import numpy as np

import habitat_sim
from habitat.sims.habitat_simulator.sim_utilities import get_ao_global_bb
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity,
    PddlSimInfo,
    SimulatorObjectType,
)
from habitat.tasks.rearrange.utils import (
    get_angle_to_pos,
    get_robot_spawns,
    rearrange_logger,
)

CAB_TYPE = "cab_type"
FRIDGE_TYPE = "fridge_type"


class ArtSampler:
    def __init__(
        self, value: float, cmp: str, override_thresh: Optional[float] = None
    ):
        self.value = value
        self.cmp = cmp
        self.override_thresh = override_thresh

    def is_satisfied(self, cur_value: float, thresh: float) -> bool:
        if self.override_thresh is not None:
            thresh = self.override_thresh

        if self.cmp == "greater":
            return cur_value > self.value - thresh
        elif self.cmp == "less":
            return cur_value < self.value + thresh
        elif self.cmp == "close":
            return abs(cur_value - self.value) < thresh
        else:
            raise ValueError(f"Unrecognized cmp {self.cmp}")

    def sample(self) -> float:
        return self.value


@dataclass
class PddlRobotState:
    """
    Specifies the configuration of the robot.

    :property place_at_pos_dist: If -1.0, this will place the robot as close
        as possible to the entity. Otherwise, it will place the robot within X
        meters of the entity.
    :property base_angle_noise: How much noise to add to the robot base angle
        when setting the robot base position.
    """

    holding: Optional[PddlEntity] = None
    should_drop: bool = False
    pos: Optional[Any] = None
    place_at_pos_dist: float = -1.0
    base_angle_noise: float = 0.0

    def sub_in(
        self, sub_dict: Dict[PddlEntity, PddlEntity]
    ) -> "PddlRobotState":
        self.holding = sub_dict.get(self.holding, self.holding)
        self.pos = sub_dict.get(self.pos, self.pos)
        return self

    def clone(self) -> "PddlRobotState":
        """
        Returns a shallow copy
        """
        return replace(self)

    def is_true(self, sim_info: PddlSimInfo, robot_entity: PddlEntity) -> bool:
        """
        Returns if the desired robot state is currently true in the simulator state.
        """
        robot_id = cast(
            int,
            sim_info.search_for_entity(robot_entity),
        )
        grasp_mgr = sim_info.sim.get_agent_data(robot_id).grasp_mgr

        assert not (self.holding is not None and self.should_drop)

        if self.holding is not None:
            # Robot must be holding desired object.
            obj_idx = cast(int, sim_info.search_for_entity(self.holding))
            abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
            if grasp_mgr.snap_idx != abs_obj_id:
                return False
        elif self.should_drop and grasp_mgr.snap_idx != None:
            return False

        if isinstance(self.pos, PddlEntity):
            targ_pos = sim_info.get_entity_pos(self.pos)
            robot = sim_info.sim.get_agent_data(robot_id).articulated_agent
            dist = np.linalg.norm(robot.base_pos - targ_pos)
            if self.place_at_pos_dist == -1.0:
                use_thresh = sim_info.robot_at_thresh
            else:
                use_thresh = self.place_at_pos_dist
            if dist > use_thresh:
                return False

        return True

    def set_state(
        self, sim_info: PddlSimInfo, robot_entity: PddlEntity
    ) -> None:
        """
        Sets the robot state in the simulator.
        """
        robot_id = cast(
            int,
            sim_info.search_for_entity(robot_entity),
        )
        sim = sim_info.sim
        grasp_mgr = sim.get_agent_data(robot_id).grasp_mgr
        # Set the snapped object information
        if self.should_drop and grasp_mgr.is_grasped:
            grasp_mgr.desnap(True)
        elif self.holding is not None:
            # Swap objects to the desired object.
            obj_idx = cast(int, sim_info.search_for_entity(self.holding))
            grasp_mgr.desnap(True)
            sim.internal_step(-1)
            grasp_mgr.snap_to_obj(sim.scene_obj_ids[obj_idx])
            sim.internal_step(-1)

        # Set the robot starting position
        if isinstance(self.pos, PddlEntity):
            targ_pos = sim_info.get_entity_pos(self.pos)
            agent = sim.get_agent_data(robot_id).articulated_agent

            if self.place_at_pos_dist == -1.0:
                # Place as close to the object as possible.
                if not sim_info.sim.is_point_within_bounds(targ_pos):
                    rearrange_logger.error(
                        f"Object {self.pos} is out of bounds but trying to set robot position"
                    )

                agent_pos = sim_info.sim.safe_snap_point(targ_pos)
                agent.base_pos = agent_pos
                agent.base_rot = get_angle_to_pos(
                    np.array(targ_pos - agent_pos)
                )
            else:
                # Place some distance away from the object.
                start_pos, start_rot, was_fail = get_robot_spawns(
                    targ_pos,
                    self.base_angle_noise,
                    self.place_at_pos_dist,
                    sim,
                    sim_info.num_spawn_attempts,
                    sim_info.physics_stability_steps,
                )
                sim.articulated_agent.base_pos = start_pos
                sim.articulated_agent.base_rot = start_rot
                if was_fail:
                    rearrange_logger.error("Failed to place the robot.")

            # We teleported the agent. We also need to teleport the object the agent was holding.
            grasp_mgr = sim.get_agent_data(robot_id).grasp_mgr
            grasp_mgr.update_object_to_grasp()

        elif self.pos is not None:
            raise ValueError(f"Unrecongized set position {self.pos}")


class PddlSimState:
    def __init__(
        self,
        art_states: Dict[PddlEntity, ArtSampler],
        obj_states: Dict[PddlEntity, PddlEntity],
        robot_states: Dict[PddlEntity, PddlRobotState],
    ):
        for k, v in obj_states.items():
            if not isinstance(k, PddlEntity) or not isinstance(v, PddlEntity):
                raise TypeError(f"Unexpected types {obj_states}")

        for k, v in art_states.items():
            if not isinstance(k, PddlEntity) or not isinstance(v, ArtSampler):
                raise TypeError(f"Unexpected types {art_states}")

        for k, v in robot_states.items():
            if not isinstance(k, PddlEntity) or not isinstance(
                v, PddlRobotState
            ):
                raise TypeError(f"Unexpected types {robot_states}")

        self._art_states = art_states
        self._obj_states = obj_states
        self._robot_states = robot_states

    def __repr__(self):
        return f"{self._art_states}, {self._obj_states}, {self._robot_states}"

    def clone(self) -> "PddlSimState":
        return PddlSimState(
            self._art_states,
            self._obj_states,
            {k: v.clone() for k, v in self._robot_states.items()},
        )

    def sub_in(self, sub_dict: Dict[PddlEntity, PddlEntity]) -> "PddlSimState":
        self._robot_states = {
            sub_dict.get(k, k): robot_state.sub_in(sub_dict)
            for k, robot_state in self._robot_states.items()
        }
        self._art_states = {
            sub_dict.get(k, k): v for k, v in self._art_states.items()
        }
        self._obj_states = {
            sub_dict.get(k, k): sub_dict.get(v, v)
            for k, v in self._obj_states.items()
        }
        return self

    def _is_object_inside(
        self, entity: PddlEntity, target: PddlEntity, sim_info: PddlSimInfo
    ) -> bool:
        """
        Returns if `entity` is inside of `target` in the CURRENT simulator state, NOT at the start of the episode.
        """
        entity_pos = sim_info.get_entity_pos(entity)
        check_marker = cast(
            MarkerInfo,
            sim_info.search_for_entity(target),
        )
        if sim_info.check_type_matches(target, FRIDGE_TYPE):
            global_bb = get_ao_global_bb(check_marker.ao_parent)
        else:
            bb = check_marker.link_node.cumulative_bb
            global_bb = habitat_sim.geo.get_transformed_bb(
                bb, check_marker.link_node.transformation
            )

        return global_bb.contains(entity_pos)

    def is_compatible(self, expr_types) -> bool:
        def type_matches(entity, match_names):
            return any(
                entity.expr_type.is_subtype_of(expr_types[match_name])
                for match_name in match_names
            )

        for entity, target in self._obj_states.items():
            if not type_matches(
                entity, [SimulatorObjectType.MOVABLE_ENTITY.value]
            ):
                return False

            if not (
                type_matches(
                    target,
                    [
                        SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value,
                        SimulatorObjectType.GOAL_ENTITY.value,
                        SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value,
                    ],
                )
            ):
                return False

            if entity.expr_type.name == target.expr_type.name:
                return False

        return all(
            type_matches(
                art_entity,
                [SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value],
            )
            for art_entity in self._art_states
        )

    def is_true(
        self,
        sim_info: PddlSimInfo,
    ) -> bool:
        """
        Returns True if the grounded state is present in the current simulator state.
        Throws exception if the arguments are not compatible.
        """

        # Check object states.
        rom = sim_info.sim.get_rigid_object_manager()
        for entity, target in self._obj_states.items():
            if not sim_info.check_type_matches(
                entity, SimulatorObjectType.MOVABLE_ENTITY.value
            ):
                raise ValueError(f"Got unexpected entity {entity}")
            obj_idx = cast(int, sim_info.search_for_entity(entity))
            abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
            entity_obj = rom.get_object_by_id(abs_obj_id)

            if sim_info.check_type_matches(
                target, SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
            ):
                # object is rigid and target is receptacle, we are checking if
                # an object is inside of a receptacle.
                if not self._is_object_inside(entity, target, sim_info):
                    return False
            elif sim_info.check_type_matches(
                target, SimulatorObjectType.GOAL_ENTITY.value
            ):
                cur_pos = entity_obj.transformation.translation

                targ_idx = cast(
                    int,
                    sim_info.search_for_entity(target),
                )
                idxs, pos_targs = sim_info.sim.get_targets()
                targ_pos = pos_targs[list(idxs).index(targ_idx)]

                dist = np.linalg.norm(cur_pos - targ_pos)
                if dist >= sim_info.obj_thresh:
                    return False
            elif sim_info.check_type_matches(
                target, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
            ):
                recep = cast(mn.Range3D, sim_info.search_for_entity(target))
                return recep.contains(entity_obj.translation)
            else:
                raise ValueError(
                    f"Got unexpected combination of {entity} and {target}"
                )

        for art_entity, set_art in self._art_states.items():
            if not sim_info.check_type_matches(
                art_entity,
                SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value,
            ):
                raise ValueError(f"Got unexpected entity {set_art}")

            marker = cast(
                MarkerInfo,
                sim_info.search_for_entity(
                    art_entity,
                ),
            )
            prev_art_pos = marker.get_targ_js()
            if not set_art.is_satisfied(prev_art_pos, sim_info.art_thresh):
                return False
        return all(
            robot_state.is_true(sim_info, robot_entity)
            for robot_entity, robot_state in self._robot_states.items()
        )

    def set_state(self, sim_info: PddlSimInfo) -> None:
        """
        Set this state in the simulator. Warning, this steps the simulator.
        """
        sim = sim_info.sim
        # Set all desired object states.
        for entity, target in self._obj_states.items():
            if not sim_info.check_type_matches(
                entity, SimulatorObjectType.MOVABLE_ENTITY.value
            ):
                raise ValueError(f"Got unexpected entity {entity}")

            if sim_info.check_type_matches(
                target, SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
            ):
                raise NotImplementedError()
            elif sim_info.check_type_matches(
                target, SimulatorObjectType.GOAL_ENTITY.value
            ):
                targ_idx = cast(
                    int,
                    sim_info.search_for_entity(target),
                )
                all_targ_idxs, pos_targs = sim.get_targets()
                targ_pos = pos_targs[list(all_targ_idxs).index(targ_idx)]
                set_T = mn.Matrix4.translation(targ_pos)
            elif sim_info.check_type_matches(
                target, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
            ):
                raise NotImplementedError()
            else:
                raise ValueError(f"Got unexpected target {target}")

            obj_idx = cast(int, sim_info.search_for_entity(entity))
            abs_obj_id = sim.scene_obj_ids[obj_idx]

            # Get the object id corresponding to this name
            rom = sim.get_rigid_object_manager()
            set_obj = rom.get_object_by_id(abs_obj_id)
            set_obj.transformation = set_T
            set_obj.angular_velocity = mn.Vector3.zero_init()
            set_obj.linear_velocity = mn.Vector3.zero_init()
            sim.internal_step(-1)
            set_obj.angular_velocity = mn.Vector3.zero_init()
            set_obj.linear_velocity = mn.Vector3.zero_init()

        # Set all desired articulated object states.
        for art_entity, set_art in self._art_states.items():
            sim = sim_info.sim
            rom = sim.get_rigid_object_manager()

            in_pred = sim_info.get_predicate("in")
            poss_entities = [
                e
                for e in sim_info.all_entities.values()
                if e.expr_type.is_subtype_of(
                    sim_info.expr_types[
                        SimulatorObjectType.MOVABLE_ENTITY.value
                    ]
                )
            ]

            move_objs = []
            for poss_entity in poss_entities:
                bound_in_pred = in_pred.clone()
                bound_in_pred.set_param_values([poss_entity, art_entity])
                if not bound_in_pred.is_true(sim_info):
                    continue
                obj_idx = cast(
                    int,
                    sim_info.search_for_entity(poss_entity),
                )
                abs_obj_id = sim.scene_obj_ids[obj_idx]
                set_obj = rom.get_object_by_id(abs_obj_id)
                move_objs.append(set_obj)

            marker = cast(
                MarkerInfo,
                sim_info.search_for_entity(
                    art_entity,
                ),
            )
            pre_link_pos = marker.link_node.transformation.translation
            marker.set_targ_js(set_art.sample())
            post_link_pos = marker.link_node.transformation.translation

            if art_entity.expr_type.is_subtype_of(
                sim_info.expr_types[CAB_TYPE]
            ):
                # Also move all objects that were in the drawer
                diff_pos = post_link_pos - pre_link_pos
                for move_obj in move_objs:
                    move_obj.translation += diff_pos

        # Set all desired robot states.
        for robot_entity, robot_state in self._robot_states.items():
            robot_state.set_state(sim_info, robot_entity)
