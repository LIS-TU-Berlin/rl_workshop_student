"""End-to-end smoke tests: run a minimal training pass and check the expected
artifacts and TensorBoard scalars come out, mirroring the manual verification
done throughout development. Not a check of learning quality (T_end is tiny),
just that the pipeline runs and produces what it's supposed to.

Run with: uv run pytest tests/
"""
import pytest
from tensorboard.backend.event_processing import event_accumulator

from scripts import train as train_module


@pytest.fixture(autouse=True)
def isolated_output_dirs(tmp_path, monkeypatch):
    """Redirect runs/ and tensorboard/ to a pytest tmp dir, so tests never touch
    real training output (see rl_workshop/paths.py)."""
    monkeypatch.setenv('RL_WORKSHOP_RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setenv('RL_WORKSHOP_TENSORBOARD_DIR', str(tmp_path / 'tensorboard'))
    return tmp_path


def _run_dir(tmp_path):
    run_dirs = list((tmp_path / 'runs').glob('*'))
    run_dirs = [d for d in run_dirs if d.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run dir, found {run_dirs}"
    return run_dirs[0]


def _scalar_tags(tmp_path):
    tb_dirs = [d for d in (tmp_path / 'tensorboard').glob('*') if d.is_dir()]
    assert len(tb_dirs) == 1, f"expected exactly one tensorboard run dir, found {tb_dirs}"
    ea = event_accumulator.EventAccumulator(str(tb_dirs[0]), size_guidance={'scalars': 0})
    ea.Reload()
    return set(ea.Tags()['scalars'])


def test_dense_reward_training_smoke(isolated_output_dirs):
    """task1: dense reward, getStartsGoals_random, no periodic eval."""
    tmp_path = isolated_output_dirs
    cfg = train_module.load_cfg('configs/disc_push_task1.yaml')
    cfg.RL.T_end = 1000
    cfg.RL.T_block = 1000
    cfg.RL.eval_episodes = 5

    train_module.main(cfg=cfg)

    run_dir = _run_dir(tmp_path)
    for fname in ['cfg.yaml', 'model.zip', 'report.csv', 'eval_episodes.csv', 'eval.mp4']:
        assert (run_dir / fname).exists(), f"missing {fname} in {run_dir}"
    assert (tmp_path / 'runs' / 'runs.csv').exists()
    assert (tmp_path / 'runs' / 'eval_runs.csv').exists()

    tags = _scalar_tags(tmp_path)
    for expected in ['train/reward', 'train/terminated', 'train/steps', 'train/reward_success']:
        assert expected in tags, f"missing tensorboard tag {expected}"
    # periodic eval wasn't enabled, so eval/* should only appear from the one final eval
    assert 'eval/success_rate' in tags


def test_sparse_reward_curriculum_and_periodic_eval_smoke(isolated_output_dirs):
    """task3: sparse reward, getStartsGoals_curriculum, periodic held-out eval enabled."""
    tmp_path = isolated_output_dirs
    cfg = train_module.load_cfg('configs/disc_push_task3.yaml')
    cfg.RL.T_end = 1000
    cfg.RL.T_block = 500
    cfg.RL.T_curr = 500
    cfg.RL.eval_episodes = 5
    cfg.TD3.periodic_eval_freq = 500
    cfg.TD3.periodic_eval_episodes = 5

    train_module.main(cfg=cfg)

    run_dir = _run_dir(tmp_path)
    for fname in ['cfg.yaml', 'model.zip', 'report.csv', 'eval_episodes.csv', 'eval.mp4']:
        assert (run_dir / fname).exists(), f"missing {fname} in {run_dir}"

    tags = _scalar_tags(tmp_path)
    for expected in ['train/reward', 'train/terminated', 'train/steps', 'eval/success_rate', 'eval/steps']:
        assert expected in tags, f"missing tensorboard tag {expected}"
