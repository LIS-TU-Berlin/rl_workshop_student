#!/bin/sh
# robotic's viewer (used even for writing a video file, not just live display) needs a
# real X server to initialize GLFW -- Xvfb is a virtual, invisible one, so nothing is ever
# actually shown on a screen. -ac disables access control so no xauth/cookie setup is
# needed for this throwaway, container-internal-only display.
set -e
Xvfb :99 -screen 0 1280x1024x24 -ac &
export DISPLAY=:99

# wait for Xvfb's socket to actually exist before continuing, rather than racing it
for _ in $(seq 1 50); do
    [ -e /tmp/.X11-unix/X99 ] && break
    sleep 0.1
done

exec "$@"
