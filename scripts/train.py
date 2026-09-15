import os
# cap BLAS threads; must happen before numpy import, so not config-driven
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('OMP_NUM_THREADS', '8')

import argparse
from typing import Optional

import numpy as np
import omegaconf, math, time
import torch
from omegaconf import DictConfig
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise
from torch.utils.tensorboard import SummaryWriter

from rl_workshop.sim_wrappers import mujoco_sb_wrappers
from rl_workshop.sim_wrappers.mujoco_gym import MujocoGym
from rl_workshop.problem_interface import ProblemInterface
from rl_workshop.telemetry.callback import CustomCallback
from rl_workshop.telemetry import eval_report
from rl_workshop.paths import runs_dir
from rl_workshop.config import load_cfg, tag_suffix


def create_RL_method(
        cfg: DictConfig,
        env: MujocoGym,
        gamma: float,
        P: Optional[ProblemInterface] = None,
        eval_env: Optional[MujocoGym] = None,
        ) -> tuple[BaseAlgorithm, CustomCallback, SummaryWriter]:
    """Sets up the SB3 RL method and the callback for logging and evaluation."""

    if cfg.RL.RL_method in ['sbTD3', 'sbTD3_her']:
        sb_env = mujoco_sb_wrappers.SB_VecGym(env)
        n_actions = env.single_action_space.shape[0]
        policy_kwargs = dict(net_arch=[256, 256])
        
        rl_method = TD3(
            policy="MlpPolicy",
            env=sb_env,
            gamma=gamma,
            #policy_delay=cfg.TD3.policy_freq,
            learning_rate=cfg.TD3.learning_rate,
            buffer_size=cfg.TD3.buffer_size,
            learning_starts=cfg.TD3.timesteps_before_training,
            batch_size=cfg.TD3.batch_size,
            gradient_steps=cfg.TD3.get('gradient_steps', 1),
            action_noise=NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=cfg.TD3.exploration_noise * np.ones(n_actions)
                ),
            target_policy_noise=cfg.TD3.target_policy_noise,
            target_noise_clip=cfg.TD3.noise_clip,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=cfg.get('seed', None),
            device=cfg.TD3.get('device', 'cpu'),
            )

        if cfg.RL.RL_method=='sbTD3_her':
            sb_env = mujoco_sb_wrappers.SB_VecGoalGym(env)
            rl_method = TD3(
                policy="MultiInputPolicy",
                env=sb_env,
                replay_buffer_class=HerReplayBuffer,
                #policy_delay=cfg.TD3.policy_freq,
                learning_rate=cfg.TD3.learning_rate, buffer_size=cfg.TD3.buffer_size,
                learning_starts=cfg.TD3.timesteps_before_training, batch_size=cfg.TD3.batch_size,
                gradient_steps=cfg.TD3.get('gradient_steps', 1),
                action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=cfg.TD3.exploration_noise * np.ones(n_actions)),
                target_policy_noise=cfg.TD3.target_policy_noise, target_noise_clip=cfg.TD3.noise_clip,
                policy_kwargs=policy_kwargs, verbose=0, seed=cfg.get('seed', None),
                device=cfg.TD3.get('device', 'cpu'),
                )
            
            rl_method.replay_buffer.copy_info_dict = True

        callback=CustomCallback(
            gym=env, 
            rl_method=rl_method, 
            tag=cfg.tag, 
            report_its=cfg.TD3.train_logging_freq,
            P=P, eval_env=eval_env,
            eval_freq_its=cfg.TD3.get('periodic_eval_freq', None),
            eval_episodes=cfg.TD3.get('periodic_eval_episodes', 20),
        )
        writer = callback.writer.writer
    else:
        raise Exception(f'RL algo not defined: {cfg.RL.RL_method}')

    return rl_method, callback, writer

def main(
        cfg: Optional[DictConfig] = None, 
        config_path: str = 'configs/disc_push.yaml') -> None:
    """Entry point for training."""

    if cfg == None:
        cfg = load_cfg(config_path)

    # cap torch CPU threads regardless of host core count
    torch.set_num_threads(cfg.TD3.get('num_threads', 4))

    # Create the problem interface
    P = ProblemInterface(cfg, tag_suffix(cfg, cfg.RL.RL_method))

    # Training env
    env = P.create_GridGym(cfg.Gym.num_scenes, 1)

    # Optional held-out evaluation env
    eval_env = None
    if cfg.TD3.get('periodic_eval_freq', None):
        eval_env = P.create_GridGym(cfg.TD3.get('periodic_eval_episodes', 20), 1)
        P.update_environment(eval_env, cfg.RL.T_end, final_evaluation=True)
        eval_env.cfg.time_limit = cfg.TD3.get('periodic_eval_time_limit', cfg.RL.eval_time_limit)

    # Create RL method
    gamma = math.pow(0.5, env.cfg.tau_step/P.cfg.RL.discount_half_life)
    rl_method, callback, writer = create_RL_method(
        cfg=cfg, 
        env=env, 
        gamma=gamma, 
        P=P, 
        eval_env=eval_env
        )

    # Training loop
    start_time = time.time()
    train_steps = 0
    while(train_steps<cfg.RL.T_end): # loops over blocks of training
        P.update_environment(env, train_steps) # to enable changes in the environment, i.e. a curriculum
        rl_method.learn(total_timesteps=env.num_scenes * cfg.RL.T_block, reset_num_timesteps=False, callback=callback)
        train_steps += cfg.RL.T_block
    train_time = time.time()-start_time

    # Final evaluation
    P.update_environment(env, train_steps, final_evaluation=True)
    succ, steps, _ = P.evaluate_policy(env, rl_method, num_episodes=cfg.RL.eval_episodes, verbose=1)

    # Logging
    writer.add_scalar('eval/success_rate', succ, train_steps+cfg.RL.eval_episodes)

    report_header = ['tag', 'scenario', 'RL_method', 'success_rate', 'avg_steps_to_success', 'train_time_sec']
    report_row = [cfg.tag, cfg.scenario, cfg.RL.RL_method, f'{succ:.3f}', f'{steps:.2f}', f'{train_time:.2f}']

    print('====', report_row)
    eval_report.append_csv_row(f'{runs_dir()}/{cfg.tag}/report.csv', report_header, report_row)
    eval_report.append_csv_row(f'{runs_dir()}/runs.csv', report_header, report_row)

    # Short evaluation video and per-episode eval results in a CSV -- opt out with
    # RL.make_video: false, e.g. in headless environments without working GLFW rendering
    if cfg.RL.get('make_video', True):
        video_env = P.create_Gym()
        P.update_environment(video_env, train_steps, final_evaluation=True)
        video_env.cfg.time_limit = cfg.RL.eval_time_limit
        video_succ, video_steps, episodes = P.evaluate_policy(
            video_env, rl_method, num_episodes=20, make_video=True, verbose=1,
        )
        eval_report.write_episode_csv(f'{runs_dir()}/{cfg.tag}/eval_episodes.csv', episodes)
        eval_report.append_csv_row(
            f'{runs_dir()}/eval_runs.csv',
            ['tag', 'scenario', 'RL_method', 'success_rate', 'avg_steps_to_success', 'num_episodes'],
            [cfg.tag, cfg.scenario, cfg.RL.RL_method, f'{video_succ:.3f}', f'{video_steps:.2f}', 20],
        )

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default='configs/disc_push.yaml')
    parser.add_argument('overrides', nargs='*', help='dotted key=value config overrides, e.g. RL.T_end=500')
    args = parser.parse_args()

    cfg = load_cfg(args.config_path)
    if args.overrides:
        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.from_dotlist(args.overrides))

    main(cfg=cfg)
