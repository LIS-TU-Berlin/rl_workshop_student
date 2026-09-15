from typing import Any, Callable, Optional, Union

from rl_workshop.sim_wrappers.mujoco_sim import MujocoSim
from rl_workshop.sim_wrappers.poly_ref import SecondOrderPolyRef
from gymnasium import Env, spaces
import numpy as np
from dataclasses import dataclass, field
import math
from enum import Enum


@dataclass
class MujocoGymCfg:
    tau_step: float = 0.05
    time_limit: float = 1.
    bounds_margin: float = .5
    goal_feat_eps: float = 1e-2
    goal_feat_points: Optional[list] = None
    w_success: float = 1.
    w_goal: float = 0.
    w_object: float = 0.
    reward_clip_dist: Optional[float] = None
    reward_robot_obj_clip_dist: Optional[float] = None
    reward_robot_point: str = 'fing'
    reward_robot_reference: str = 'obj_base'
    w_stop: float = 0.
    w_effort: float = 0.
    w_hit: float = 0.
    goal_feat_vel_eps: Optional[float] = None
    reward_fct_name: Optional[str] = None
    feat_dims: Optional[list] = None
    action_scale: float = 1.
    eff_action: Optional[str] = None
    observation_points: Optional[list] = None
    obs_pos_scale: float = 1.
    obs_pos_offset: list = field(default_factory=list)
    obs_vel_scale: float = .1
    obs_referr_scale: float = 10.

class MujocoGym(Env):
    metadata: dict = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode: str = 'human'
    verbose: int = 0

    def __init__(
        self,
        sim: MujocoSim,
        cfg: MujocoGymCfg,
        num_scenes: int = 1,
        terminal_bounds: Optional[np.ndarray] = None,
    ) -> None:
        self.sim = sim

        self.cfg = cfg
        if self.cfg.eff_action=='none':
            self.cfg.eff_action=None
        if self.cfg.goal_feat_points==[]:
            self.cfg.goal_feat_points=None
        if self.cfg.observation_points==[]:
            self.cfg.observation_points=None
        if self.cfg.obs_pos_offset==[]:
            self.cfg.obs_pos_offset=None

        self.feat_dims = list(self.cfg.feat_dims) if self._cfg_get('feat_dims', None) is not None else [0, 1, 2]

        self.terminal_bounds = terminal_bounds
        self.num_scenes = num_scenes
        self.num_worlds = 1
        if self.sim.warp_worlds>0.:
            self.num_worlds = self.sim.warp_worlds
            self.num_scenes *= self.sim.warp_worlds
            self.terminal_bounds = np.tile(self.terminal_bounds, (self.num_worlds, 1, 1))
            self.terminal_bounds = np.transpose(self.terminal_bounds, (1, 0, 2))
            self.terminal_bounds = np.reshape(self.terminal_bounds, (2, -1)) #(2, s*nq)
        self.scene_needs_reset = np.ones((self.num_scenes), dtype=bool) #(s)
        self.scene_time = np.zeros((self.num_scenes)) #(s)
        self.ctrl_indices = self.sim.ctrl_indices.reshape(self.num_scenes//self.num_worlds, -1)[0] #(nu)
        self.ctrl_current = self.qpos[:, self.ctrl_indices].copy() #(ns, nu)

        # before evaluating observations and goal features, we grab body_ids for the 
        # observation/goal_points as well as potential eff_action control
        if self.cfg.goal_feat_points is not None:
            self.goal_feat_pt_ids = np.array(self.sim.get_pt_ids(self.cfg.goal_feat_points), dtype=np.int32)
            self.goal_feat_pt_ids = self.goal_feat_pt_ids.reshape(len(self.cfg.goal_feat_points), num_scenes).transpose() #(ns, npt)
        if self.cfg.observation_points is not None:
            # observation_points is a list of point-groups; within each group, the last point
            # is the reference and gets subtracted from the others (see get_pt_feat)
            self.observation_pt_groups = []
            for group in self.cfg.observation_points:
                ids = np.array(self.sim.get_pt_ids(group), dtype=np.int32)
                ids = ids.reshape(len(group), num_scenes).transpose() #(ns, npt)
                self.observation_pt_groups.append(ids)
        if self.cfg.eff_action is not None:
            self.eff_pt_ids = np.array(self.sim.get_pt_ids([self.cfg.eff_action]), dtype=np.int32)
            self.eff_pt_ids = self.eff_pt_ids.reshape(1, num_scenes).transpose() #(ns, 1)

        # to get the first observation, we need to setup a ctrlRef, get a goal feature, then query an observation
        self.goal_feat = np.zeros((self.num_scenes, len(self.feat_dims)*(self.goal_feat_pt_ids.shape[1]-1))) #the last is the reference!
        self.goal_q = self.qpos # just to initialize somehow
        self.observation, self.feat, obs_info = self.observation_fct(self.qpos, self.qvel, self.ctrl_current, self.goal_feat, info_string=True)
        self.observation_space = spaces.Box(-2., +2., shape=self.observation.shape, dtype=np.float32)
        self.single_observation_space = spaces.Box(-2., +2., shape=(self.observation.shape[1],), dtype=np.float32)

        # define the action space
        self.action_scale = self.cfg.action_scale*math.sqrt(self.cfg.tau_step)
        if self.cfg.eff_action is None:
            self.action_dim = self.sim.model.nu // (self.num_scenes//self.num_worlds)
        else:
            self.action_dim = 3 #6
            self.q_home = self.qpos[:, self.ctrl_indices]
        self.action_space = spaces.Box(-1., +1., shape=(self.num_scenes, self.action_dim), dtype=np.float32)
        self.single_action_space = spaces.Box(-1., +1., shape=(self.action_dim,), dtype=np.float32)

        # a counter to loop through provided starts/goals
        self.starts_goals_counter = 0

        # for logging
        self.total_eps = 0
        self.total_steps = 0
        self.total_reward = 0
        self.total_terminated = 0
        self.total_reward_terms = {} #per-component reward sums, see reward_fct's self.reward_terms
        
        print(f"-- initialized MjGym with observation {obs_info}, action dim {self.action_dim} (pose_action={self.cfg.eff_action}), tau step {self.cfg.tau_step}, and time limit {self.cfg.time_limit}")

    def set_starts_goals(
        self,
        starts_q: np.ndarray,
        starts_r: np.ndarray,
        goals_q: np.ndarray,
        goals_feat: np.ndarray,
    ) -> None:
        assert starts_q.shape[0]==goals_q.shape[0]
        self.starts_goals_counter = 0
        self.starts_q = np.atleast_2d(starts_q)
        self.starts_v = np.zeros((starts_q.shape[0], self.sim.model.nv//(self.num_scenes//self.num_worlds)))
        self.starts_r = np.atleast_2d(starts_r)
        self.goals_q = np.atleast_2d(goals_q)
        # self.goals_v = np.zeros((goals_q.shape[0], self.sim.qvel_dim//(self.num_scenes//self.num_worlds)))
        self.goals_feat = np.atleast_2d(goals_feat)

    def next_starts_goals_counter(self) -> int:
        if self.verbose > 0 and self.starts_goals_counter > self.starts_q.shape[0]:
            print(f'-- WARNING: reusing start_goals: counter:{self.starts_goals_counter} data:{self.starts_q.shape[0]}')
        i = self.starts_goals_counter % self.starts_q.shape[0]
        self.starts_goals_counter += 1
        return i

    def auto_reset(self) -> tuple[np.ndarray, dict]:
        assert self.num_scenes>0
        qpos, qvel, act = self.qpos, self.qvel, self.act
        needs_set = False
        
        # overwrite qpos, qvel, act, self.ctrl_current
        for s in range(self.num_scenes):
            if self.scene_needs_reset[s]:
                i = self.next_starts_goals_counter()

                self.goal_feat[s] = self.goals_feat[i]
                self.goal_q[s] = self.goals_q[i:i+1]
                # self.goal_feat[s] = self.goal_feat_map(self.goals_q[i:i+1]) #, self.goals_v[i:i+1])
                qpos[s] = self.starts_q[i]
                qvel[s] = self.starts_v[i]
                act[s] *= 0.
                # self.ctrl_current[s] = qpos[s:s+1, self.ctrl_indices]
                self.ctrl_current[s] = self.starts_r[i, self.ctrl_indices]
                
                needs_set = True
                
        # set state, including mj_forward!, which is needed for observations
        if needs_set:
            self.sim.set_state(qpos, qvel, act)

        # get observations
        for s in range(self.num_scenes):
            if self.scene_needs_reset[s]:
                self.observation[s], self.feat[s] = self.observation_fct(qpos[s:s+1], qvel[s:s+1], self.ctrl_current[s:s+1], self.goal_feat[s:s+1], selected_scenes=[s])

                self.scene_needs_reset[s] = False
                self.scene_time[s] = 0.

        return self.observation, { 'feature': self.feat }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """resets all scenes in the env"""
        if seed is not None:
            super().reset(seed=int(seed))

        self.scene_needs_reset[:] = True
        
        return self.auto_reset()    

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        if action.ndim==1:
            action = action.reshape(self.num_scenes, -1)
        assert action.shape[0]==self.num_scenes
        if self.cfg.eff_action is None:
            assert action.shape[1]==len(self.ctrl_indices)
        else:
            assert action.shape[1]==3 #6

        # is a endeffector delta action?
        if self.cfg.eff_action is not None:
            action = self.convert_eff_action(action)

        robot_obj_dist_prev = None
        if self._cfg_float('w_object', 0.) != 0.:
            robot_obj_dist_prev = self.robot_obj_distance(self.feat)

        # set action
        action_delta = self.action_scale * action
        current_vel = self.qvel[:, self.ctrl_indices]
        tmp = SecondOrderPolyRef(self.sim.ctrl_time, self.ctrl_current, current_vel, action_delta, 2.*self.cfg.tau_step)
        self.sim.ctrl_buffer = tmp.sample_buffer(self.sim.ctrl_time,
                                                    self.sim.ctrl_time+self.cfg.tau_step,
                                                    self.sim.tau_sim)
        self.sim.ctrl_bufferPtr = 0

        # step
        self.sim.step(tau_step=self.cfg.tau_step)
        self.scene_time += self.cfg.tau_step
        self.ctrl_current = self.sim.ctrl_buffer[self.sim.ctrl_bufferPtr]
  
        # get obs and truncation
        feat_prev = self.feat
        self.observation, self.feat = self.observation_fct(self.qpos, self.qvel, self.ctrl_current, self.goal_feat)
        obj_vel = self.qvel[:, -6:-3] # object is the last free body in qpos/qvel
        reward_fct_name = self._cfg_get('reward_fct_name', None)
        reward_fn = getattr(self, reward_fct_name) if reward_fct_name else self.reward_fct
        reward, terminated = reward_fn(self.observation, self.feat, feat_prev, self.goal_feat, robot_obj_dist_prev, action, obj_vel)
        for k, v in self.reward_terms.items():
            self.total_reward_terms[k] = self.total_reward_terms.get(k, 0.) + np.sum(v)

        # terminated = self.is_goal(self.feat)
        truncated = (self.scene_time >= self.cfg.time_limit) # terminated and truncated difference is super important
        if self.terminal_bounds is not None:
            truncated |= self.is_out_of_bound(self.qpos, self.qvel)
        self.scene_needs_reset = np.logical_or(terminated, truncated)

        self.total_eps += np.count_nonzero(self.scene_needs_reset)
        self.total_steps += self.num_scenes
        self.total_reward += np.sum(reward)
        self.total_terminated += np.count_nonzero(terminated)

        if self.num_scenes==1: # for stable_baselines to work..
            reward = reward.item()
        return self.observation, reward, terminated, truncated, { 'feature': self.feat }

    def _cfg_get(self, key: str, default: Any) -> Any:
        if hasattr(self.cfg, 'get'):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def _cfg_float(self, key: str, default: Optional[float]) -> float:
        value = self._cfg_get(key, default)
        if value is None:
            return default
        return float(value)

    def clipped_delta_reward(self, dist_prev: np.ndarray, dist: np.ndarray, clip_dist: float) -> np.ndarray:
        # "closeness" ramps linearly from 1 at dist=0 down to 0 at dist=clip_dist, clamped at 0 beyond that
        close_prev = np.maximum(1. - dist_prev / clip_dist, np.zeros(dist.shape))
        close_now = np.maximum(1. - dist / clip_dist, np.zeros(dist.shape))
        return close_now - close_prev

    def robot_obj_distance(self, feat: np.ndarray) -> np.ndarray:
        if feat.shape[1] != len(self.feat_dims):
            raise ValueError(f'robot-object reward expects a single {len(self.feat_dims)}-D goal feature')

        if not hasattr(self, 'reward_robot_pt_ids'):
            robot_point = self._cfg_get('reward_robot_point', 'fing')
            reference_point = self._cfg_get('reward_robot_reference', 'obj_base')
            self.reward_robot_pt_ids = np.array(self.sim.get_pt_ids([robot_point, reference_point]), dtype=np.int32)
            self.reward_robot_pt_ids = self.reward_robot_pt_ids.reshape(2, self.num_scenes).transpose()

        robot_feat = self.get_pt_feat(self.reward_robot_pt_ids)
        return self.feat_distance(robot_feat, feat)

    def reward_fct_sparse(
        self,
        obs: np.ndarray,
        feat: np.ndarray,
        feat_prev: np.ndarray,
        goal_feat: np.ndarray,
        robot_obj_dist_prev: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        obj_vel: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sparse reward, fixed goal, no curriculum. 
        When the object feature is within goal_feat_eps of the goal, the reward is 1.0 else zero. 

        Args:
            obs: Current observation, shape (num_scenes, obs_dim).
            feat: Current object goal-feature, shape (num_scenes, feat_dim).
            feat_prev: Previous-step object goal-feature, shape (num_scenes, feat_dim).
            goal_feat: Target goal-feature to compare against, shape (num_scenes, feat_dim).
            robot_obj_dist_prev: Previous robot-object distance, shape (num_scenes,).
            action: Action taken this step, shape (num_scenes, action_dim).
            obj_vel: Object velocity, shape (num_scenes, 3).

        Returns:
            Tuple of (reward, is_goal), each shape (num_scenes,).
        """
        # Sparse success reward
        dist = self.feat_distance(feat, goal_feat)
        is_goal = (dist <= self.cfg.goal_feat_eps)
        success_reward = self._cfg_float('w_success', 1.) * np.where(is_goal, 1., 0.)

        self.reward_terms = {'success': success_reward}
        return success_reward, is_goal

    def reward_fct_dense1(
        self,
        obs: np.ndarray,
        feat: np.ndarray,
        feat_prev: np.ndarray,
        goal_feat: np.ndarray,
        robot_obj_dist_prev: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        obj_vel: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dense reward, fixed goal, no curriculum.
        Combines a binary success term with reward-shaping.

        Args:
            obs: Current observation, shape (num_scenes, obs_dim).
            feat: Current object goal-feature, shape (num_scenes, feat_dim).
            feat_prev: Previous-step object goal-feature, shape (num_scenes, feat_dim).
            goal_feat: Target goal-feature to compare against, shape (num_scenes, feat_dim).
            robot_obj_dist_prev: Previous robot-object distance, shape (num_scenes,).
            action: Action taken this step, shape (num_scenes, action_dim).
            obj_vel: Object velocity, shape (num_scenes, 3). Remains unused here.

        Returns:
            Tuple of (reward, is_goal), each shape (num_scenes,).
        """

        # Sparse success reward
        dist = self.feat_distance(feat, goal_feat)
        is_goal = (dist <= self.cfg.goal_feat_eps)
        success_reward = self._cfg_float('w_success', 1.) * np.where(is_goal, 1., 0.)

        # Dense goal reward: rewards a decrease in (clipped) distance to the goal
        dist_prev = self.feat_distance(feat_prev, goal_feat)
        clip_dist = self._cfg_float('reward_clip_dist', 10. * self.cfg.goal_feat_eps)
        ### BEGIN IMPLEMENTATION
        # Compute reward_goal using dist_prev, dist, and clip_dist
        # Refer to slide 16 for help
        raise NotImplementedError("reward_fct_dense1 incomplete!") 
        ### END IMPLEMENTATION

        # Dense object reward: rewards the robot decreasing its (clipped) distance to the object
        robot_obj_dist = self.robot_obj_distance(feat)
        robot_obj_clip_dist = self._cfg_float('reward_robot_obj_clip_dist', 10. * self.cfg.goal_feat_eps)
        ### BEGIN IMPLEMENTATION
        # Compute reward_object using robot_obj_dist_prev, robot_obj_dist, robot_obj_clip_dist
        # Refer to slide 16 for help
        raise NotImplementedError("reward_fct_dense1 incomplete!")
        ### END IMPLEMENTATION

        # Penalizes action magnitude
        #effort_reward = -self._cfg_float('w_effort', 0.) * np.mean(np.square(action), axis=1)

        self.reward_terms = {
            'success': success_reward,
            'goal': reward_goal,
            'object': reward_object,
            #'effort': effort_reward,
        }
        r = success_reward + reward_goal + reward_object #+ effort_reward
        return r, is_goal

    def reward_fct_dense2(
        self,
        obs: np.ndarray,
        feat: np.ndarray,
        feat_prev: np.ndarray,
        goal_feat: np.ndarray,
        robot_obj_dist_prev: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        obj_vel: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dense reward, fixed goal, no curriculum, "hit, don't push" shaping.
        On top of reward_fct_dense1's terms, rewards the object's velocity component toward the goal,
        so a single decisive hit that sends the object gliding to the goal scores higher
        than staying in continuous contact and walking it there with many small corrective
        pushes (set w_hit > 0 to enable; 0 reduces this to reward_fct_dense1's reward).

        Args:
            obs: Current observation, shape (num_scenes, obs_dim).
            feat: Current object goal-feature, shape (num_scenes, feat_dim).
            feat_prev: Previous-step object goal-feature, shape (num_scenes, feat_dim).
            goal_feat: Target goal-feature to compare against, shape (num_scenes, feat_dim).
            robot_obj_dist_prev: Previous robot-object distance, shape (num_scenes,).
            action: Action taken this step, shape (num_scenes, action_dim).
            obj_vel: Object velocity, shape (num_scenes, 3).

        Returns:
            Tuple of (reward, is_goal), each shape (num_scenes,).
        """

        # Sparse success reward
        dist = self.feat_distance(feat, goal_feat)
        is_goal = (dist <= self.cfg.goal_feat_eps)
        success_reward = self._cfg_float('w_success', 1.) * np.where(is_goal, 1., 0.)

        # Dense goal reward: rewards a decrease in (clipped) distance to the goal
        dist_prev = self.feat_distance(feat_prev, goal_feat)
        clip_dist = self._cfg_float('reward_clip_dist', 10. * self.cfg.goal_feat_eps)
        ### BEGIN IMPLEMENTATION
        # Compute reward_goal using dist_prev, dist, and clip_dist
        # Refer to slide 18 for help
        raise NotImplementedError("reward_fct_dense2 incomplete!")
        ### END IMPLEMENTATION

        # Dense object reward: rewards the robot decreasing its (clipped) distance to the object
        robot_obj_dist = self.robot_obj_distance(feat)
        robot_obj_clip_dist = self._cfg_float('reward_robot_obj_clip_dist', 10. * self.cfg.goal_feat_eps)
        ### BEGIN IMPLEMENTATION
        # Compute reward_object using robot_obj_dist_prev, robot_obj_dist, robot_obj_clip_dist
        # Refer to slide 18 for help
        raise NotImplementedError("reward_fct_dense2 incomplete!")
        ### END IMPLEMENTATION


        # Penalizes action magnitude
        # effort_reward = -self._cfg_float('w_effort', 0.) * np.mean(np.square(action), axis=1)

        # Hit reward: rewards the object's velocity component toward the goal (positive only,
        # so standing still isn't penalized that's already covered by goal/success).
        # A quick strike that sends the object flying toward the goal scores high here
        hit_scale = self._cfg_float('w_hit', 0.)

        if hit_scale != 0.:
            if obj_vel is None:
                raise ValueError('obj_vel is required when hit reward is enabled')
            goal_dir = (goal_feat - feat) / np.maximum(dist[:, None], 1e-6)
            # Dot product of object velocity with the normalized direction to the goal
            vel_toward_goal = np.sum(obj_vel[:, self.feat_dims] * goal_dir, axis=1)

            ### BEGIN IMPLEMENTATION
            # Compute hit_reward using hit_scale (variable that is set to w_hit) and vel_toward_goal (result of the dot product above)
            # Hint: Use np.maximum to make sure that the dimension of the reward is (num_scenes,)
            # Refer to slide 18 for help
            raise NotImplementedError("reward_fct_dense2 incomplete!")
            ### END IMPLEMENTATION
        else:
            hit_reward = np.zeros(dist.shape)

        self.reward_terms = {
            'success': success_reward,
            'goal': reward_goal,
            'object': reward_object,
            # 'effort': effort_reward,
            'hit': hit_reward,
        }
        r = success_reward + reward_goal + reward_object + hit_reward #+ effort_reward
        return r, is_goal

        
    def observation_fct(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        cref: np.ndarray,
        goal_feat: np.ndarray,
        selected_scenes: Optional[list] = None,
        info_string: bool = False,
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, str]]:
        """
        Composes the observation from the raw mujoco data.
        Concatenates robot joint position, full joint velocity, and control tracking error,
        optionally with extra 3D observation points, then the goal feature relative to the
        object's current position. Result is clipped to [-10, 10].

        NOTE: the method needs to be vectorized, returning the matrix of observations for all scenes in the env.

        Args:
            qpos: Joint positions for all scenes, shape (num_scenes, nq).
            qvel: Joint velocities for all scenes, shape (num_scenes, nv).
            cref: Reference (target) control positions, shape (num_scenes, len(ctrl_indices)).
            goal_feat: Target goal-feature for the episode, shape (num_scenes, feat_dim).
            selected_scenes: Optional subset of scene indices to compute observation points for.
            info_string: If True, also return a string reporting each observation block's width.

        Returns:
            Tuple of (obs, feat) or (obs, feat, info) if info_string is True.
            obs: Observation, shape (num_scenes, obs_dim).
            feat: Object's current goal-feature, shape (num_scenes, feat_dim).
        """

        # Joint space observations
        # this excludes the object
        o_pos = self.cfg.obs_pos_scale * qpos[:, self.ctrl_indices] 

        if self.cfg.obs_pos_offset is not None:
            o_pos -= self.cfg.obs_pos_scale * np.array(self.cfg.obs_pos_offset)

        # This INCLUDES! all object velocities, including angular
        o_vel = self.cfg.obs_vel_scale * qvel 

        # Control tracking error: target minus current joint position
        o_err = self.cfg.obs_referr_scale * (cref - qpos[:, self.ctrl_indices]) 

        if self.cfg.eff_action is not None:
            sel = selected_scenes if selected_scenes is not None else range(self.num_scenes)
            Jpos, _ = self.sim.get_Jacobian(self.eff_pt_ids[sel])
            nc = len(self.ctrl_indices)
            ns = len(sel)
            Jpos = Jpos.reshape((ns, 3, self.num_scenes, -1))
            Jpos = Jpos[:,:,sel,:]
            Jpos = np.diagonal(Jpos, axis1=0, axis2=2)
            Jpos = np.transpose(Jpos, (2,0,1))
            Jpos = Jpos[:,:,:nc]
            o_err = Jpos * o_err[:, None, :]
            o_err = np.sum(o_err, axis=2)

        if info_string:
            info = f'#scenes:{o_pos.shape[0]} qpos:{o_pos.shape[1]} qvel:{o_vel.shape[1]} err:{o_err.shape[1]}'

        # Stack all the observations together
        obs = np.hstack((o_pos, o_vel, o_err))

        # Point observations (each group referenced to its own last point, see __init__)
        if self.cfg.observation_points is not None:
            for ids in self.observation_pt_groups:
                o_pts = self.cfg.obs_pos_scale * self.get_pt_feat(ids, selected_scenes)
                obs = np.hstack((obs, o_pts))
                if info_string:
                    info += f' pts:{o_pts.shape[1]}'

        # Goal feature observation
        if self.cfg.goal_feat_points is not None:
            feat = self.get_pt_feat(self.goal_feat_pt_ids, selected_scenes)
            # feat2 = self.goal_feat_map(qpos) #, qvel) # for debugging
            # print(np.linalg.norm(feat-feat2))
            assert goal_feat.shape==feat.shape
            if info_string:
                info += f' goal_pts:{feat.shape[1]}'
        else:
            raise Exception('deprecated: only point goal features supported right now')
            # feat = self.goal_feat_map(qpos) #, qvel) #AVOID THIS!! too slow
            if info_string:
                info += f' goal_map:{feat.shape[1]}'
                
        o_goal = self.cfg.obs_pos_scale * (goal_feat - feat) #we observe the goal RELATIVE to current
        obs = np.hstack((obs, o_goal))

        # Final clipping
        obs = np.clip(obs, -10., 10.)
        if info_string:
            info += f' total:{obs.shape[1]}'
            return obs, feat, info
        
        return obs, feat

    # Needed to support HER
    def apply_delta_goal_feat_to_observation(
        self, org_obs: np.ndarray, org_goal_feat: np.ndarray, new_goal_feat: np.ndarray
    ) -> np.ndarray:
        assert org_obs.ndim==2 and org_goal_feat.ndim==1 and new_goal_feat.ndim==1
        nf = org_goal_feat.shape[0]
        obs = org_obs.copy()
        obs[:, -nf:] += self.cfg.obs_pos_scale * (new_goal_feat - org_goal_feat)
        return obs
    
    # Needed to support HER
    def get_feat_from_observation(self, obs: np.ndarray, goal_feat: np.ndarray) -> np.ndarray:
        assert obs.ndim==1 and goal_feat.ndim==1
        nf = goal_feat.shape[0]
        return goal_feat - (obs[-nf:]/self.cfg.obs_pos_scale)
        
    # Helper to compute point observations
    def get_pt_feat(self, pt_ids: np.ndarray, selected_scenes: Optional[list] = None) -> np.ndarray:
        if selected_scenes is None:
            pts = self.sim.get_pts(pt_ids)
            # the last is the reference!
            pts -= pts[:,-1:,:]
            pts = pts[:,:-1,:]
            pts = pts[:,:,self.feat_dims]
            return pts.reshape(pt_ids.shape[0], len(self.feat_dims)*(pt_ids.shape[1]-1))
        else:
            pts = self.sim.get_pts(pt_ids[selected_scenes])
            # the last is the reference!
            pts -= pts[:,-1:,:]
            pts = pts[:,:-1,:]
            pts = pts[:,:,self.feat_dims]
            return pts.reshape(len(selected_scenes), len(self.feat_dims)*(pt_ids.shape[1]-1))

    def feat_distance(self, feat: np.ndarray, goal_feat: np.ndarray) -> np.ndarray:
        return np.linalg.norm(feat-goal_feat, axis=1)

    # Helper to check if the robot is out of bounds
    def is_out_of_bound(self, qpos: np.ndarray, qvel: np.ndarray) -> Union[bool, np.ndarray]:
        if self.terminal_bounds is None or self.cfg.bounds_margin<0.:
            return False
        assert self.terminal_bounds.shape[0]==2
        assert self.terminal_bounds.shape[1]==qpos.size
        assert not np.any(self.terminal_bounds[1]<=self.terminal_bounds[0]), f"bounds (joint) not proper: {self.terminal_bounds}"
        l = np.any(qpos < self.terminal_bounds[0].reshape(qpos.shape) - self.cfg.bounds_margin, axis=1)
        g = np.any(qpos > self.terminal_bounds[1].reshape(qpos.shape) + self.cfg.bounds_margin, axis=1)
#        if qpos.shape[0]==1 and (l[0] or g[0]): #for debugging in single scene setting: which bound is violated
#            j_names = np.array(self.sim.C.getJointNames())
#            assert j_names.size==qpos.shape[1]
#            print(j_names)
#            if l[0]:
#                err = ((self.terminal_bounds[0] - self.cfg.bounds_margin) - qpos).reshape(-1)
#                print('-- low bound violation:', j_names[np.where(err>0.)], err)
#            if g[0]:
#                err = (qpos - (self.terminal_bounds[1] + self.cfg.bounds_margin)).reshape(-1)
#                idx = np.where(err>0.)
#                print('--  up bound violation:', j_names[idx], err)
        return np.logical_or(l,g) 

    # Helper to convert end-effector actions
    def convert_eff_action(self, action: np.ndarray, pos_only: bool = True) -> np.ndarray:
        ns = self.num_scenes
        nc = len(self.ctrl_indices)
        pose_action = action
        action = np.zeros((ns, nc))
        for s in range(ns):
            Jpos, Jang = self.sim.get_Jacobian(self.eff_pt_ids[s])
            if pos_only:
                J = Jpos[0]
                J = J.reshape(3, ns, -1)
            else:
                J = np.vstack((Jpos[0], Jang[0]))
                J = J.reshape(6, ns, -1)
            J = J[:,s,:nc]
            Jinv = J.T @ np.linalg.pinv(J@J.T+1e-3*np.eye(J.shape[0]))
            action[s,:] = Jinv @ pose_action[s,:]
            if self.q_home is not None:
                action[s,:] += 0.1*(np.eye(nc) - Jinv@J) @ (self.q_home[s, :] - self.qpos[s, :nc])
        return action
    
    ### minimal example to evaluate a policy

    def rollout(self, pi: Callable, return_data: bool = False) -> Optional[dict]:
        '''helper to play and view a policy'''

        obs, info = self.reset()

        if return_data:
            data = {'state': [], 'obs': [], 'action': [], 'ctrl_cost': [], 'next_obs': [], 'reward': [], 'terminal': []}

        t = 0
        R = 0
        while True:
            action = pi(obs, t)
            self.sim.ctrl_costs=0.
            next_obs, reward, terminated, truncated, info = self.step(action)
            if return_data:
                data['obs'].append(obs)
                data['action'].append(action)
                data['ctrl_cost'].append(self.sim.ctrl_costs)
                data['reward'].append(np.array([reward]))
                data['next_obs'].append(next_obs)
                data['terminal'].append(np.array([(1 if terminated else 0)], dtype=np.int16))
            obs = next_obs
            R += reward
            t += 1
            if self.verbose>1:
                print("reward: ", reward)
            if np.any(np.logical_or(terminated, truncated)):
                break

        if self.verbose>0:
            print('total (non-discounted) return:', R)
        
        if return_data:
            for key, value in data.items():
                data[key] = np.stack(value)
            return data

    @property
    def qpos(self) -> np.ndarray:
        return self.sim._qpos.reshape(self.num_scenes, -1)

    @property
    def qvel(self) -> np.ndarray:
        return self.sim.data.qvel.reshape(self.num_scenes, -1)

    @property
    def act(self) -> np.ndarray:
        return self.sim.data.actuator_force.reshape(self.num_scenes, -1)

    def render(self) -> Optional[np.ndarray]:
        '''also part of the env.Gym'''
        self.C.view(False, f'RoboticGym time {self.time} / {self.cfg.time_limit}')
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()
