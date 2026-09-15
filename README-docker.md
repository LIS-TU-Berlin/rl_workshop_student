# RL Tutorial on Dense and Sparse Rewards (Docker)

## Installation

### Windows / macOS Docker installation

**Prerequisites**:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose) or [Docker Engine (on Linux)](https://docs.docker.com/engine/install)


**Installation**:
1. Clone the repository and `cd` into it.
```bash
git clone https://github.com/LIS-TU-Berlin/rl_workshop_student.git
cd rl_workshop_student
```
2. Get the image, either by building it yourself or downloading a pre-built copy:

   **Option A -- build it yourself**.
   ```bash
   docker compose build
   ```

   **Option B -- download a pre-built image**.
   ```bash
   # Download rl-workshop-image.tar https://tubcloud.tu-berlin.de/s/ZHDX4mmnpwEHSBy and place in rl_workshop_student
   docker load -i rl-workshop-image.tar
   ```

   Then, either way, start the container
   ```bash
   docker compose run --rm --service-ports workshop
   ```

3. Verify the installation
```bash
uv run pytest tests/test_docker_smoke.py
```

## Training and Evaluation

Rendering is unreliable in Docker, so videos are turned off.

### 1. Run a trained policy

```bash
# uv run python scripts/eval.py <tag> --no-video
uv run python scripts/eval.py 260910-190202-disc_push-task4-sbTD3-seed100 --no-video
```

### 2. Train a naive policy
```bash
# Trains using the sparse reward only
uv run scripts/train.py configs/disc_push_task0.yaml RL.make_video=false
```

### 3. Train using a dense reward function
```bash
# Uses reward_fct_dense1() and a fixed goal
uv run scripts/train.py configs/disc_push_task1.yaml RL.make_video=false

# Uses reward_fct_dense2() and a fixed goal
uv run scripts/train.py configs/disc_push_task2.yaml RL.make_video=false
```

### 4. Train using a sparse reward function with state sampling/curriculum
```bash
# Random goal, with finger initialized near object
uv run scripts/train.py configs/disc_push_task3.yaml RL.make_video=false

# Random goal, with finger initialized randomly
uv run scripts/train.py configs/disc_push_task4.yaml RL.make_video=false
```

### 3. View Tensorboard logs
To view logs while training is still running (e.g. from step 2/3/4 above) in another
terminal, open a new terminal window, `cd` into the repo, and attach a second shell to the
already-running container instead of starting a new one:
```bash
docker compose exec workshop bash
```
Then, in that shell:
```bash
# --host 0.0.0.0 is needed in the container so it's reachable from your host browser
uv run tensorboard --logdir tensorboard --host 0.0.0.0
```
Then open [http://localhost:6006](http://localhost:6006) in your host browser.
