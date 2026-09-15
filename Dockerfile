# CPU-only workshop image. Exists mainly to work around `robotic` (the rai/ry bindings)
# only publishing Linux x86_64 wheels on PyPI -- no macOS or Windows wheels, so native
# install (see README) doesn't work there. This gives everyone the same Linux environment.
#
# Ubuntu 24.04 ships Python 3.12.3 by default, matching this project's Python version, and
# matches the OS `robotic`'s own install docs are written for (apt package names below are
# taken from https://marctoussaint.github.io/robotic/getting_started.html).
FROM ubuntu:24.04

# freeglut3/libglew-dev/liblapack3 are the exact packages the robotic docs list; the rest
# is extra rendering-backend coverage (osmesa/EGL/GLFW) and xvfb -- robotic's viewer (used
# even just to write a video file, not only for live display) needs a real X server to
# initialize GLFW, so docker-entrypoint.sh starts Xvfb (virtual, nothing ever shown on a
# real screen) automatically for every command run in this container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    freeglut3-dev \
    libglew-dev \
    liblapack3 \
    libblas3 \
    libgfortran5 \
    libgl1 \
    libglu1-mesa \
    libglfw3 \
    libosmesa6 \
    libegl1 \
    libgles2 \
    libgl1-mesa-dri \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# grab the uv/uvx binaries directly from astral's image, rather than curl-installing them
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# headless software rendering backend for MuJoCo (no GPU needed/available in the container)
ENV MUJOCO_GL=osmesa

# force Mesa to use its CPU software rasterizer (llvmpipe) instead of probing for a real
# GPU driver, which doesn't exist in this container -- Xvfb alone isn't enough for GLX/
# OpenGL context creation, this is needed too (see docker-entrypoint.sh)
ENV LIBGL_ALWAYS_SOFTWARE=1

# Venv lives outside /workspace on purpose: docker-compose bind-mounts the host repo over
# /workspace, and a venv placed at the default /workspace/.venv would collide with (and risk
# being overwritten through) the host's own .venv at the same path.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace

# install dependencies first, in their own layer, so `uv sync` is only re-run (and
# re-downloads nothing) when pyproject.toml/uv.lock actually change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# so a plain `python`/interactive shell also sees the venv, not just `uv run ...`
ENV PATH="/opt/venv/bin:${PATH}"

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
