# RL Tutorial on Dense and Sparse Rewards

## Installation

### Linux native installation

**Prerequisites**: 
- Python 3.10-3.12 [[link](https://www.python.org/downloads/)]
- git [[link](https://git-scm.com/install/)]
- uv [[link](https://docs.astral.sh/uv/getting-started/installation/)]

**Installation**:
1. Clone the repository and `cd` into it.
```bash
git clone https://github.com/LIS-TU-Berlin/rl_workshop_student.git
cd rl_workshop_student
```
2. Run `uv sync` to create a `.venv/` and install all dependencies pinned in `uv.lock`.
```bash
uv sync
```
3. Verify the installation
```bash
uv run pytest tests/test_installation_smoke.py
```

### Docker

See [README-docker.md](README-docker.md) for the Docker-based setup.

## Training and Evaluation

### 1. Run a trained policy

```bash
# uv run python scripts/eval.py <tag>
uv run python scripts/eval.py 260910-190202-disc_push-task4-sbTD3-seed100
```

### 2. Train a naive policy
```bash
# Trains using the sparse reward only
uv run scripts/train.py configs/disc_push_task0.yaml RL.T_end=3000
```

### 3. Train using a dense reward function
```bash
# Uses reward_fct_dense1() and a fixed goal
uv run scripts/train.py configs/disc_push_task1.yaml

# Uses reward_fct_dense2() and a fixed goal
uv run scripts/train.py configs/disc_push_task2.yaml
```

### 4. Train using a sparse reward function with state sampling/curriculum
```bash
# Random goal, with finger initialized near object
uv run scripts/train.py configs/disc_push_task3.yaml

# Random goal, with finger initialized randomly
uv run scripts/train.py configs/disc_push_task4.yaml
```

### 3. View Tensorboard logs
```bash
tensorboard --logdir tensorboard
```

## Description of important scripts

- `scripts/train.py`: Train a new policy. Trained policy and other artifacts are created under `runs/`. Tensorboard logs are created in `tensorboard/`

- `scripts/eval.py`: Evaluate a trained policy using a tag (e.g. `260910-174709-disc_push-task3-sbTD3-seed100`) from the `runs/` folder.

- `rl_workshop/problem_interface.py`: Builds the simulated scene from a config and wraps it for RL training, and generates the start/goal positions each episode uses.

- `rl_workshop/sim_wrappers/mujoco_gym.py`: Defines MujocoGym, a [Gymnasium](https://gymnasium.farama.org/)  environment (Gymnasium is the standard RL API expected by RL libraries) for our task. Rather than simulating one scene at a time, it tiles `num_scenes` copies of the scene into a single MuJoCo model and steps them all together in one batched call so that every observation, reward, and action carries a `num_scenes` dimension, giving much faster data collection than running scenes one by one. 

- `configs/*.yaml`: Settings used for training. `configs/disc_push.yaml` defines the base configuration, and `configs/disc_push_task<n>.yaml` adds overrides for task `n`.

- `scenes/disc_push.yml`: Defines the simulated scene used for training.


## CPU threads

Training/eval cap CPU threads (PyTorch + BLAS) to 8 by default, regardless of how many cores the host has. To use a different value for one run, set the env vars before invoking the script:

```
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 uv run python scripts/train.py configs/disc_push_task3.yaml
```
