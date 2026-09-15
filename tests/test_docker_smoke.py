"""Same pre-implementation sanity check as test_installation_smoke.py, but with video
generation disabled (RL.make_video: false). Rendering (env.sim.C.view(...), via GLFW) needs
a working OpenGL context, which has proven flaky in some headless Docker/Xvfb setups even
with a virtual display configured -- this test isolates "does the actual RL pipeline work"
from "does GLFW rendering work in this particular container".

Run with: uv run pytest tests/test_docker_smoke.py
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


def test_docker_smoke(isolated_output_dirs):
    """task0: sparse reward, getStartsGoals_random -- fully implemented, no student code needed."""
    tmp_path = isolated_output_dirs
    cfg = train_module.load_cfg('configs/disc_push_task0.yaml')
    cfg.RL.T_end = 1000
    cfg.RL.T_block = 1000
    cfg.RL.eval_episodes = 5
    cfg.RL.make_video = False

    train_module.main(cfg=cfg)

    run_dirs = [d for d in (tmp_path / 'runs').glob('*') if d.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run dir, found {run_dirs}"
    run_dir = run_dirs[0]
    for fname in ['cfg.yaml', 'model.zip', 'report.csv']:
        assert (run_dir / fname).exists(), f"missing {fname} in {run_dir}"

    tb_dirs = [d for d in (tmp_path / 'tensorboard').glob('*') if d.is_dir()]
    assert len(tb_dirs) == 1, f"expected exactly one tensorboard run dir, found {tb_dirs}"
    ea = event_accumulator.EventAccumulator(str(tb_dirs[0]), size_guidance={'scalars': 0})
    ea.Reload()
    tags = set(ea.Tags()['scalars'])
    for expected in ['train/reward', 'train/terminated', 'train/steps', 'eval/success_rate']:
        assert expected in tags, f"missing tensorboard tag {expected}"
