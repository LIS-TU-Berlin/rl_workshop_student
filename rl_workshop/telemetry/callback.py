from typing import Optional

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from rl_workshop.sim_wrappers import mujoco_gym
from rl_workshop.telemetry.writer import Writer
from rl_workshop.paths import runs_dir
from rl_workshop.problem_interface import ProblemInterface


class CustomCallback(BaseCallback):
    """SB3 training callback: periodically reports training-rollout stats to
    TensorBoard/checkpoints (train/*), and optionally runs held-out
    evaluation rollouts at a coarser frequency (eval/*)."""

    def __init__(
            self,
            gym: mujoco_gym.MujocoGym,
            rl_method: BaseAlgorithm,
            tag: str,
            report_its: int = 100,
            P: Optional[ProblemInterface] = None,
            eval_env: Optional[mujoco_gym.MujocoGym] = None,
            eval_freq_its: Optional[int] = None,
            eval_episodes: int = 20,
            verbose: int = 0
            ):
        """
        gym: training env
        rl_method: the SB3 algorithm being trained
        tag: run tag, used for both the TensorBoard log dir and checkpoint path
        report_its: how often (in training iterations) to log train/* stats and save a checkpoint
        P: ProblemInterface used to run held-out evaluation rollouts, required if eval_env is set
        eval_env: separate held-out env to evaluate on, periodic eval is disabled if None
        eval_freq_its: how often (in training iterations) to run a held-out eval
        eval_episodes: number of episodes per held-out eval
        verbose: passed through to BaseCallback
        """
        super().__init__(verbose)
        self.gym = gym
        self.it = 0
        self.report_its = report_its
        self.next_report = report_its
        self.writer = Writer(tag)
        self.rl_method = rl_method
        self.P = P
        self.eval_env = eval_env
        self.eval_freq_its = eval_freq_its
        self.eval_episodes = eval_episodes
        self.next_eval = eval_freq_its

    def _on_step(self) -> bool:
        """Called by SB3 after every environment step. Logs train/* stats and checkpoints
        the model every `report_its` iterations; runs a held-out eval and logs eval/* every
        `eval_freq_its` iterations, if enabled."""
        self.it += 1
        if self.it>=self.next_report and self.gym.total_eps>0:
            print(f'-- it:{self.it}, t:{self.gym.total_steps}, #eps:{self.gym.total_eps}, total_reward:{self.gym.total_reward}')
            self.writer.add("train/reward", self.gym.total_reward / self.gym.total_eps)
            self.writer.add("train/terminated", self.gym.total_terminated / self.gym.total_eps)
            self.writer.add("train/steps", self.gym.total_steps / self.gym.total_eps)

            # If reward has components, log each component separately as well
            for key, value in self.gym.total_reward_terms.items():
                self.writer.add(f"train/reward_{key}", value / self.gym.total_eps)

            self.gym.total_eps, self.gym.total_steps, self.gym.total_reward, self.gym.total_terminated = 0, 0, 0, 0
            self.gym.total_reward_terms = {}
            self.next_report += self.report_its

            self.writer.write(self.it, verbose=1)

            self.rl_method.save(f'{runs_dir()}/{self.writer.tag}/model.zip')

        # Held-out evaluation
        if self.eval_env is not None and self.eval_freq_its and self.it>=self.next_eval:
            self.next_eval += self.eval_freq_its
            succ, _steps_on_success, episodes = self.P.evaluate_policy(
                env=self.eval_env, 
                policy=self.rl_method, 
                num_episodes=self.eval_episodes, 
                verbose=0,
            )
            avg_steps = sum(ep['steps'] for ep in episodes) / len(episodes)
            print(f'-- held-out eval at it:{self.it}: success_rate={succ:.3f}, avg_steps={avg_steps:.2f}')
            self.writer.writer.add_scalar('eval/success_rate', succ, self.it)
            self.writer.writer.add_scalar('eval/steps', avg_steps, self.it)
            self.writer.writer.flush()

        return True
