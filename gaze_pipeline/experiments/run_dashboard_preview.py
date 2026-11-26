#!/usr/bin/env python3

"""Preview the local dashboard without running the robot."""

import time

from gaze_pipeline.runtime.dashboard import (
    DashboardState,
    LocalDashboardServer,
)


state = DashboardState()
state.update(
    "runtime",
    phase="WAITING",
    message="Dashboard preview — robot runtime not connected.",
)

server = LocalDashboardServer(state)

try:
    url = server.start()
    print(f"Visitor screen: {url}/visitor")
    print(f"Research screen: {url}/research")
    print("Press Ctrl+C to stop.")

    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping dashboard preview.")
finally:
    server.stop()
