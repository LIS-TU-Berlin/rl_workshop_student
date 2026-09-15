import os


def runs_dir():
    """Base directory for run artifacts (config, checkpoints, reports), normally 'runs'.
    Override with RL_WORKSHOP_RUNS_DIR, e.g. to keep smoke-test output out of the shared
    runs/ directory used by real training runs."""
    return os.environ.get('RL_WORKSHOP_RUNS_DIR', 'runs')


def tensorboard_dir():
    """Base directory for TensorBoard logs, normally 'tensorboard'. Override with
    RL_WORKSHOP_TENSORBOARD_DIR, e.g. to keep smoke-test output out of the shared
    tensorboard/ directory used by real training runs."""
    return os.environ.get('RL_WORKSHOP_TENSORBOARD_DIR', 'tensorboard')
