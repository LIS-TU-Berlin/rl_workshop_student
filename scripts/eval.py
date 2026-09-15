"""Load and run a previously trained policy checkpoint by its run tag, or play a
uniform-random policy for visually sanity-checking an env/config with no checkpoint.

Each training run writes its checkpoint and config under runs/<tag>/ (see
ProblemInterface and CustomCallback). This script rebuilds the same
environment from the saved config and replays the saved model in it.

Find a tag by listing the runs/ directory, or checking runs/runs.csv (one line
per completed run).

Writes one row per evaluated episode (episode, success, reward, steps) to
runs/<tag>-replay/eval_episodes.csv, and appends a summary row (tag, scenario,
RL_method, success rate, avg steps, episode count) to runs/eval_runs.csv.

By default, records a video (runs/<tag>-replay/eval.mp4), per RL.make_video in the run's
config; override with --video/--no-video (--no-video skips rendering entirely, just
computing success/steps stats -- no display of any kind needed). --live instead shows an
interactive live view (needs a real display).

Usage:
    uv run python scripts/eval.py <tag>
    uv run python scripts/eval.py <tag> --episodes 20 --no-video
    uv run python scripts/eval.py --random [--config configs/disc_push.yaml]
    uv run python scripts/eval.py --show-starts-goals [--config configs/disc_push.yaml]
"""
import os
# cap BLAS threads before numpy import (see scripts/train.py)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('OMP_NUM_THREADS', '8')

import argparse
from typing import Optional

import numpy as np
import omegaconf
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_workshop.config import load_cfg, tag_suffix
from rl_workshop.problem_interface import ProblemInterface
from rl_workshop.sim_wrappers.mujoco_gym import MujocoGym
from rl_workshop.telemetry import eval_report
from rl_workshop.paths import runs_dir as default_runs_dir


def play_random_policy(config_path: str = 'configs/disc_push.yaml', num_rollouts: int = 10) -> None:
    """Rolls out a uniform-random policy for visually sanity-checking an env/config --
    no trained checkpoint involved."""
    np.random.seed(0)
    cfg = load_cfg(config_path)
    P = ProblemInterface(cfg, tag_suffix(cfg, cfg.RL.RL_method))
    env = P.create_Gym()
    P.update_environment(env, 0, True)
    pi = lambda x, t: 1. * np.random.randn(env.num_scenes, env.action_dim)
    env.sim.view_speed = 1.
    env.cfg.time_limit = 2.
    for _ in range(num_rollouts):
        env.rollout(pi, False)


def display_random_starts_goals(config_path: str = 'configs/disc_push.yaml') -> None:
    """Displays random start-goal pairs for a given config."""
    cfg = load_cfg(config_path)
    P = ProblemInterface(cfg, tag_suffix(cfg, cfg.RL.RL_method))
    starts_q, starts_r, goals_q, goals_feat = P.getStartsGoals_curriculum(10, 1.)
    n = starts_q.shape[0]
    for i in range(n):
        P.C.setJointState(goals_q[i])
        goal_pose = P.C.getFrame('obj').getPose()
        P.C.setJointState(starts_q[i])
        P.C.getFrame('obj_goal').setPose(goal_pose)
        P.C.view(True, f'start {i}')


def load_policy(tag: str, runs_dir: Optional[str] = None) -> tuple[ProblemInterface, MujocoGym, BaseAlgorithm]:
    run_dir = os.path.join(runs_dir or default_runs_dir(), tag)
    cfg_path = os.path.join(run_dir, 'cfg.yaml')
    checkpoint_path = os.path.join(run_dir, 'model.zip')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"no saved config for tag '{tag}': {cfg_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"no saved checkpoint for tag '{tag}': {checkpoint_path}")

    cfg = omegaconf.OmegaConf.load(cfg_path)
    torch.set_num_threads(cfg.TD3.get('num_threads', 4))
    P = ProblemInterface(cfg, f'-{tag}-replay')
    env = P.create_Gym()
    P.update_environment(env, cfg.RL.T_end, final_evaluation=True)
    env.cfg.time_limit = cfg.RL.eval_time_limit

    policy = TD3.load(checkpoint_path, device=cfg.TD3.get('device', 'cpu'))
    return P, env, policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tag', nargs='?', help="run tag, e.g. 260908-130324-disc_push-scenario2-sbTD3")
    parser.add_argument('--random', action='store_true', help='play a uniform-random policy instead of loading a checkpoint (ignores tag)')
    parser.add_argument('--show-starts-goals', action='store_true', help='display random start-goal pairs instead of loading a checkpoint (ignores tag)')
    parser.add_argument('--config', default='configs/disc_push.yaml', help='config to use with --random or --show-starts-goals')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--video', dest='video', action='store_true', default=None,
                         help='record a video (runs/<tag>-replay/eval.mp4) (default: RL.make_video from the run config)')
    parser.add_argument('--no-video', dest='video', action='store_false',
                         help="don't render at all -- just compute success/steps stats")
    parser.add_argument('--live', action='store_true',
                         help='view live instead of recording a video or running headless (needs a real display)')
    parser.add_argument('--verbose', type=int, default=2)
    args = parser.parse_args()

    if args.random:
        play_random_policy(args.config)
        return

    if args.show_starts_goals:
        display_random_starts_goals(args.config)
        return

    if not args.tag:
        parser.error('tag is required unless --random or --show-starts-goals is given')

    P, env, policy = load_policy(args.tag)
    if args.live:
        view, make_video = True, False
    else:
        view = False
        make_video = args.video if args.video is not None else P.cfg.RL.get('make_video', True)
    succ, steps, episodes = P.evaluate_policy(
        env, policy, num_episodes=args.episodes,
        view=view, make_video=make_video, verbose=args.verbose,
    )
    print(f'== success rate: {succ:.3f}, avg steps to success: {steps:.2f}')

    eval_report.write_episode_csv(f'{P.run_dir}/eval_episodes.csv', episodes)
    eval_report.append_csv_row(
        f'{default_runs_dir()}/eval_runs.csv',
        ['tag', 'scenario', 'RL_method', 'success_rate', 'avg_steps_to_success', 'num_episodes'],
        [args.tag, P.cfg.scenario, P.cfg.RL.RL_method, f'{succ:.3f}', f'{steps:.2f}', args.episodes],
    )


if __name__ == '__main__':
    main()
