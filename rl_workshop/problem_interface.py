from typing import Union
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_workshop.sim_wrappers import mujoco_io, mujoco_gym, mujoco_sim
from rl_workshop.telemetry import video_gen
from rl_workshop.paths import runs_dir
import robotic as ry
import numpy as np
from datetime import datetime
import omegaconf
import os, time, math


class ProblemInterface:
    """
    This is a problem specific class, which loads a configuration and provides
    helpers to create a (parallelized) sim and gym.
    It also provides stable-baselines compatible wrappers.
    Crucially, it also provides random starts and goals.
    """

    object_frame_name = "obj"
    goal_frame_name = "obj_goal"

    def __init__(self, cfg: DictConfig, tag_append: str = ""):

        # Process the cfg (overwrite scenario-specific settings, then save)
        self.cfg = cfg
        self.cfg.tag = f'{datetime.now().strftime("%y%m%d-%H%M%S")}{tag_append}'

        print("-- creating Workspace for scenario", self.cfg.scenario)

        self.run_dir = f"{runs_dir()}/{self.cfg.tag}"
        os.system(f"mkdir -p {self.run_dir}")
        omegaconf.OmegaConf.save(config=self.cfg, f=f"{self.run_dir}/cfg.yaml")

        # Load the robotic configuration
        self.C = ry.Config()
        self.C.addFile(self.cfg.files.scene_yaml)
        self._prepare_scene()
        self.q_home = self.C.getJointState()
        self.f_obj = self.C.getFrame(self.object_frame_name)
        assert (
            self.f_obj is not None
        ), f"cannot find task object frame {self.object_frame_name}"
        self.obj_rest_z = self.f_obj.getPosition()[
            2
        ]  # the scene-authored resting height, before any randomization
        self.f_manip = (
            self.C.getFrame(self.cfg.Gym.reward_robot_point)
            if self.cfg.Gym.get("reward_robot_point", None)
            else None
        )
        self._finalize_scene()

        # Grab the goal feature pts, which define the goal feature map, and thereby the goal-conditioning in Gym
        self.goal_feat_pts_frames = []
        if self.cfg.Gym.goal_feat_points is not None:
            for name in self.cfg.Gym.goal_feat_points:
                f = self.C.getFrame(name)
                assert f is not None, f"cannot find goal_feat_pts frame {name}"
                self.goal_feat_pts_frames.append(f)

        print(
            "-- loaded config q-dim:",
            self.C.getJointDimension(),
            "with joints",
            self.C.getJointNames(),
        )

    def _prepare_scene(self) -> None:
        """Hook for task-specific scene normalization before joint state is cached."""
        pass

    def _finalize_scene(self) -> None:
        """Hook for task state that depends on finalized joint indexing."""
        pass

    def create_grid_config(self, nx: int, ny: int, dist: float) -> ry.Config:
        self.Cgrid = ry.Config()
        for x in range(nx):
            for y in range(ny):
                base = self.Cgrid.addConfigurationCopy(
                    self.C, f"{x*ny+y}_"
                )  # the string adds a prefix to all frame names
                for i in range(base.ID, self.Cgrid.getFrameDimension()):
                    f = self.Cgrid.frame(i)
                    if f.getParent() is None:
                        p = f.getPosition()
                        p[0] += dist * (x - (nx - 1) / 2)
                        p[1] += dist * (y - (ny - 1) / 2)
                        f.setPosition(p)
        for f in self.Cgrid.getFrames():
            if "camera_init" in f.name:
                f.name = "camera_init"
                p = f.getPosition()
                p[0] += dist * (nx - 1) / 2
                p[1] += dist * (ny - 1) / 2
                f.setPosition(p)
                break

        return self.Cgrid

    def get_xml(self, C: ry.Config) -> str:
        solref = self.cfg.Gym.get("solref", None) or "0.002 1."
        M = mujoco_io.MujocoWriter(C, friction=self.cfg.Gym.friction, solref=solref)
        xml = M.str().decode("ascii")
        with open("z.xml", "w") as fil:
            fil.write(xml)
        return xml

    def create_Sim(self, engine: str = "mujoco") -> Union[mujoco_sim.MujocoSim, ry.Simulation]:
        if engine == "mujoco":
            xml = self.get_xml(self.C)
            sim = mujoco_sim.MujocoSim(
                xml,
                self.C,
                tau_sim=self.cfg.Gym.tau_sim,
                warp_worlds=self.cfg.Gym.warp_worlds,
                use_mj_viewer=self.cfg.Gym.view_mj,
            )
        elif engine == "physx":
            sim = ry.Simulation(
                self.C, engine=ry.SimulationEngine.physx, verbose=0
            )
        else:
            raise Exception(f'engine "{engine}" not defined')
        return sim

    def create_Gym(self, num_scenes: int = 1) -> mujoco_gym.MujocoGym:
        sim = self.create_Sim()

        return mujoco_gym.MujocoGym(
            sim,
            cfg=self.cfg.Gym,
            num_scenes=num_scenes,
            terminal_bounds=sim.C.getJointLimits(),
        )

    def create_GridGym(self, nx: int, ny: int) -> mujoco_gym.MujocoGym:
        Cgrid = self.create_grid_config(nx, ny, 1.5)
        C_org = self.C

        self.C = Cgrid
        gym = self.create_Gym(nx * ny)

        self.C = C_org
        return gym

    # This is awkward: we duplicate implementation of computing goal_features:
    # In the Gym environment this is done by evaluating the mj_sim ...
    # but here we do it based on a ry-Config. But they need to be consistent!
    def goal_feat_map(self, qpos: np.ndarray) -> np.ndarray:
        assert qpos.ndim == 1

        feat_dims = (
            list(self.cfg.Gym.feat_dims)
            if self.cfg.Gym.get("feat_dims", None) is not None
            else [0, 1, 2]
        )

        q0 = self.C.getJointState()
        self.C.setJointState(qpos)
        feat = np.empty((len(self.goal_feat_pts_frames), 3))
        for i, f in enumerate(self.goal_feat_pts_frames):
            feat[i] = f.getPosition()
        # the last is the reference!
        for i in range(feat.shape[0]):
            feat[i] -= feat[-1]
        self.C.setJointState(q0)
        return feat[:-1][:, feat_dims].reshape(-1)

    def update_environment(
        self,
        env: mujoco_gym.MujocoGym,
        train_steps: int,
        final_evaluation: bool = False
        ) -> None:
        # phase
        alpha = (train_steps + self.cfg.RL.T_block) / (self.cfg.RL.T_curr + 1)
        alpha = min(alpha, 1.0)
        env.cfg.time_limit = 2.0 * alpha * self.cfg.RL.nominal_time_limit

        # how many start/goals samples for this block?
        if final_evaluation:  # final evaluation
            n = self.cfg.RL.eval_episodes
        elif env.starts_goals_counter == 0:  # first block
             # heuristic: enough goals if episodes would last only 10% of nominal time limit
            n = (self.cfg.RL.T_block * env.num_scenes / (0.1 * self.cfg.RL.nominal_time_limit / env.cfg.tau_step)) 
            n = int(min(n, 1e6))
        else:
            # heuristic: twice as many goals as needed in the previous block
            n = (2 * env.starts_goals_counter)  

        # sample new start/goals for this block
        starts_goals_fct_name = self.cfg.RL.get("starts_goals_fct_name", "getStartsGoals_random")
        starts_goals_fn = getattr(self, starts_goals_fct_name)
        starts, starts_r, goals, goals_feat = starts_goals_fn(n=n, alpha=alpha, final_evaluation=final_evaluation)
        env.set_starts_goals(starts, starts_r, goals, goals_feat)

        print(
            f"=== t: {train_steps}, alpha: {alpha}, starts/goals: {env.starts_q.shape[0]}, time_limit: {env.cfg.time_limit}"
        )

    def getStartsGoals_random(
        self, n: int, alpha: float, final_evaluation: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fully random starts and goals, no curriculum (alpha is accepted but unused, so this
        has the same call signature as getStartsGoals_curriculum and can be selected via
        RL.starts_goals_fct_name interchangeably). Used for the dense-reward tasks.

        Args:
            n: Number of start/goal pairs to sample.
            alpha: Unused; accepted only to keep the same signature as getStartsGoals_curriculum.
            final_evaluation: Unused; accepted only to keep the same signature as
                getStartsGoals_curriculum (there's no curriculum here to bypass).

        Returns:
            Tuple of (starts_q, starts_r, goals_q, goals_feat), each stacked to shape (n, ...).
        """
        # fixed_goal: the simplest task -- goal stays at the scene-authored default pose
        # (e.g. table center) every episode; only the start is randomized. Random-goal variants
        # (e.g. HER) leave this false and resample a new goal every episode instead.
        fixed_goal = self.cfg.RL.get("fixed_goal", False)
        if fixed_goal:
            q_goal_fixed = self.q_home.copy()
            goal_feat_fixed = self.goal_feat_map(q_goal_fixed)

        starts_q, starts_r, goals_q, goals_feat = [], [], [], []
        for _ in range(n):
            self.set_random_config()
            q_start = self.C.getJointState()
            if fixed_goal:
                q_goal = q_goal_fixed
            else:
                self.set_random_config()
                q_goal = self.C.getJointState()
            starts_q.append(q_start)
            starts_r.append(q_start)
            goals_q.append(q_goal)
            goals_feat.append(
                goal_feat_fixed if fixed_goal else self.goal_feat_map(q_goal)
            )

        return (
            np.stack(starts_q),
            np.stack(starts_r),
            np.stack(goals_q),
            np.stack(goals_feat),
        )

    def getStartsGoals_curriculum(
        self, n: int, alpha: float, final_evaluation: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Starts interpolated toward the goal by a curriculum, ramping from near the
        goal (alpha near 0) to fully random (alpha=1) as alpha increases over training (see
        update_environment). Used for the sparse-reward task, where there's no shaping
        gradient far from the goal, so the policy needs easy near-goal resets early on to
        ever stumble into a success worth learning from.

        Args:
            n: Number of start/goal pairs to sample.
            alpha: Current curriculum ceiling in (0, 1], from update_environment.
            final_evaluation: If True, bypass the curriculum entirely (t=1, fully decoupled
                start/goal) instead of the usual t ~ U(0, alpha). Independent evaluation
                (held-out during-training checks, and the final end-of-training eval) should
                always measure true full difficulty, not a mix of difficulties diluted by
                whatever curriculum ceiling happens to be in effect -- otherwise curriculum and
                no-curriculum runs wouldn't be evaluated under comparable conditions.

        Returns:
            Tuple of (starts_q, starts_r, goals_q, goals_feat), each stacked to shape (n, ...).
            - starts_q: full joint config the scene resets to.
            - starts_r: initial control reference (currently = starts_q).
            - goals_q: full joint config at the goal.
            - goals_feat: object's target x,y (relative to obj_base), derived from goals_q.

        Note: the manipulator's start joints are placed near the object's (interpolated) start,
        offset by min_start_sep at a random angle. RL.finger_start_mode picks how that offset
        behaves across the curriculum - 'near': always kept, even at t=1 (easier); 'random':
        tapers with t to the independent q_start_random draw at t=1 (full difficulty).
        """
        assert alpha > 0.0 and alpha <= 1.0

        fixed_goal = self.cfg.RL.get("fixed_goal", False)
        if fixed_goal:
            q_goal_fixed = self.q_home.copy()
            goal_feat_fixed = self.goal_feat_map(q_goal_fixed)

        starts_q, starts_r, goals_q, goals_feat = [], [], [], []
        
        for _ in range(n):

            # interpolation between goal and start
            self.set_random_config()
            q_start_random = self.C.getJointState()
            if fixed_goal:
                q_goal = q_goal_fixed
            else:
                self.set_random_config()
                q_goal = self.C.getJointState()

            # randomized interpolation factor within the curriculum's current ceiling alpha
            # (t ~ U[0, alpha], as in arXiv:2602.08557 eq. 9): gives a spread of difficulties
            # up to the current curriculum limit. For final_evaluation, always use t=1 (fully
            # decoupled start/goal, no interpolation) instead - see docstring.

            ### BEGIN IMPLEMENTATION
            # Refer to slide 23 for help
            raise NotImplementedError("Implement the interpolation and curriculum logic here")
            ### END IMPLEMENTATION

            # manipulator start: offset from the object's start (never coincident); near/random
            # picks whether that offset holds fixed or tapers to a random draw at t=1
            n_robot = q_start.shape[0] - 7
            min_sep = self.cfg.Gym.get("min_start_sep", None) or 0.0
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            offset = min_sep * np.array([np.cos(theta), np.sin(theta)])

            # Decide where to initialize the finger
            # near - at a fixed radial offset from the object and random angle (easier)
            # random - randomly and independently sampled (harder)
            near_pos = q_start[-7:-7 + n_robot] + offset[:n_robot]
            finger_start_mode = self.cfg.RL.get("finger_start_mode", "near")
            if finger_start_mode == "near":
                q_start[:n_robot] = near_pos
            elif finger_start_mode == "random":
                q_start[:n_robot] = (1.0 - t) * near_pos + t * q_start_random[:n_robot]
            else:
                raise ValueError(f"unknown RL.finger_start_mode: {finger_start_mode!r}")

            starts_q.append(q_start)
            starts_r.append(q_start)
            goals_q.append(q_goal)
            goals_feat.append(
                goal_feat_fixed if fixed_goal else self.goal_feat_map(q_goal)
            )

        return (
            np.stack(starts_q),
            np.stack(starts_r),
            np.stack(goals_q),
            np.stack(goals_feat),
        )

    def set_random_config(self) -> None:
        # q = self.q_home.copy()
        # self.C.setJointState(q)
        # pos = self.f_obj.getPosition()
        # sigma = .05
        # q[:2] += sigma * np.random.randn(2)
        # pos[:2] += sigma * np.random.randn(2)
        # self.C.setJointState(q)
        # self.f_obj.setPosition(pos)

        # Uniform randomization
        min_sep = self.cfg.Gym.get("min_start_sep", None)
        for _ in range(20):
            self.C.setRandom()
            pos = self.f_obj.getPosition()
            pos[2] = self.obj_rest_z
            self.f_obj.setPosition(pos)
            # setRandom() always samples a true-uniform SO(3) rotation for free joints,
            # ignoring any box `limits` given for the quaternion dims -- so a flat disc/puck
            # object needs its orientation force-reset to upright here explicitly.
            if self.cfg.Gym.get("obj_upright", False):
                self.f_obj.setQuaternion([1, 0, 0, 0])
            if min_sep is None or self.f_manip is None:
                break
            # reject/resample configs where the manipulator starts overlapping the object
            # (independent uniform sampling of both otherwise regularly produces interpenetrating
            # starts, which are physically invalid and can make contact resolution unstable)
            sep = np.linalg.norm(
                np.array(self.f_manip.getPosition()[:2]) - np.array(pos[:2])
            )
            if sep >= min_sep:
                break

    def evaluate_policy(
        self,
        env: mujoco_gym.MujocoGym,
        policy: BaseAlgorithm,
        num_episodes: int = 20,
        view: bool = False,
        make_video: bool = False,
        verbose: int = 2,
        add_perturbations: bool = False,
    ) -> tuple[float, float, list[dict]]:
        assert not (
            view and make_video
        ), "cannot view and make video at the same time currently"

        print("== evaluating policy...")

        video_path = f"{self.run_dir}/eval.mp4"

        F_goals, F_obj = [], []
        for f in env.sim.C.getFrames():
            if f.name.endswith(self.goal_frame_name):
                F_goals.append(f)
            if f.name.endswith(self.object_frame_name):
                F_obj.append(f)

        if view:
            env.sim.view_speed = 1.0
        if make_video:
            env.sim.C.view(False)
            env.sim.view_speed = 1.0
            env.sim.saved_images = []
            rgb = env.sim.C.get_viewer().getRgb()
            V = video_gen.VideoGenerator(input_framerate=50, filename=video_path)
            V.add([rgb] * 10)

        n_ep = 0
        n_succ = 0.0
        steps_succ = 0.0
        steps_succ_sqr = 0.0
        ep_reward = np.zeros((env.num_scenes))
        ep_steps = np.zeros((env.num_scenes))
        episodes = []

        while n_ep < num_episodes:
            state, info = env.auto_reset()
            if env.num_scenes == 1:
                env.sim.view_text = (
                    f"episode {n_ep}, acc. reward {ep_reward.item():.3f},"
                )

            # reposition goal frames for illustration
            goal_glob = env.goal_q.copy()
            for th in range(len(F_goals)):
                F_goals[th].setRelativePosition(goal_glob[th, -7:-4])
                F_goals[th].setRelativeQuaternion(goal_glob[th, -4:])

            # add perturbation?
            for f in F_obj:
                f.setColor([1.0, 0.5, 0.0])
            if add_perturbations and (n_ep // 10) % 2:
                env.sim.view_text += " WITH PERTURBATIONS (RED),"
                if np.random.random() < 0.2:
                    delta = np.random.normal(env.num_scenes, 3)
                    env.qvel[:, -6:-3] += 0.5 / np.linalg.norm(delta) * delta
                    for f in F_obj:
                        f.setColor([1.0, 0.0, 0.0])

            # get action - account for HER replay buffer observations..
            if hasattr(policy, "replay_buffer") and hasattr(
                policy.replay_buffer, "her_ratio"
            ):
                obs_dict = {
                    "observation": env.observation,
                    "achieved_goal": env.feat,
                    "desired_goal": env.goal_feat,
                }
                action, _ = policy.predict(obs_dict, deterministic=True)
            else:
                action, _ = policy.predict(state, deterministic=True)

            if verbose > 2:
                print(ep_steps, f"{env.sim.ctrl_time:.3f}", state, action)

            # step
            state, reward, terminated, truncated, info = env.step(action)
            ep_finished = np.logical_or(terminated, truncated)
            ep_reward += reward
            ep_steps += 1.0

            for k in range(env.num_scenes):
                if ep_finished[k]:
                    if verbose > 1:
                        print(
                            f"episode {n_ep}, terminated: {terminated[k]}, acc. reward: {ep_reward[k]} steps: {ep_steps[k]}"
                        )

                    if terminated[k]:
                        if ep_reward[k] < 0.999:
                            print(
                                "-- WARNING: success should be equivalent to termimated and equivalent to eq_reward>1."
                            )
                        if view:
                            if env.num_scenes == 1:
                                env.sim.C.view(
                                    False,
                                    f"episode {n_ep}, acc. reward {ep_reward.item():.3f}, simulation t: {env.sim.ctrl_time:6.3f} -- SUCCESS",
                                )
                            # time.sleep(1.)
                        n_succ += 1
                        steps_succ += ep_steps[k]
                        steps_succ_sqr += ep_steps[k] * ep_steps[k]

                    if make_video:
                        if terminated[k] and ep_steps[k] >= 5:
                            env.sim.C.view(
                                False,
                                f"episode {n_ep}, acc. reward {ep_reward.item():.3f}, simulation t: {env.sim.ctrl_time:6.3f} -- SUCCESS",
                                offscreen=True,
                            )
                            rgb = env.sim.C.get_viewer().getRgb()
                            env.sim.saved_images.append([rgb] * 10)
                        V.add(env.sim.saved_images)
                        env.sim.saved_images = []

                    episodes.append({
                        "episode": n_ep,
                        "success": int(terminated[k]),
                        "reward": float(ep_reward[k]),
                        "steps": int(ep_steps[k]),
                    })

                    ep_reward[k] = 0.0
                    ep_steps[k] = 0.0
                    n_ep += 1

                    if (n_ep % 10) == 0:
                        print("\r   episode: ", n_ep, "/", num_episodes, end="")

        if n_succ > 0:
            steps_succ /= n_succ
            steps_succ_sqr /= n_succ
            len_std = math.sqrt(steps_succ_sqr - steps_succ * steps_succ)
        else:
            len_std = 0.0
        print(
            f"\n== #ep: {n_ep}  succ_rate: {n_succ/n_ep}  avg succ length: {steps_succ}+={len_std}"
        )
        if make_video:
            print(f"-- video written to {video_path}")

        return n_succ / n_ep, steps_succ, episodes
