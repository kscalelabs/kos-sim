"""Test script for the viewer.

This script is used to test the viewer.

Usage:

python -m kos_sim.test_viewer
"""

import asyncio
import math

import mujoco
import mujoco_viewer
from kscale import K

api = K()

# Gets the base path.
# zbot_dir = asyncio.run(api.download_and_extract_urdf("zbot-v2"))
# zbot_mjcf = next(zbot_dir.glob("*.mjcf"))

zbot_mjcf = "/Users/ben/Github/urdfs/bots/kbot/gpr-20241204.mjcf"

model = mujoco.MjModel.from_xml_path(str(zbot_mjcf))
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# Get simulation timestep
timestep = model.opt.timestep
last_update = asyncio.get_event_loop().time()
start_time = last_update

mujoco.mj_step(model, data)

# simulate and render
for _ in range(100000):
    if viewer.is_alive:
        current_time = asyncio.get_event_loop().time()
        sim_time = current_time - last_update

        # Calculate elapsed time since start for sinusoidal motion
        elapsed_time = current_time - start_time

        # Apply sinusoidal motion to each joint
        # Adjust amplitude and frequency as needed
        amplitude = 0.5  # radians
        frequency = 0.5  # Hz
        for i in range(model.nu):
            data.ctrl[i] = amplitude * math.sin(2 * math.pi * frequency * elapsed_time)

        # Step the simulation to match real time
        # while sim_time > 0:
        #     mujoco.mj_step(model, data)
        #     sim_time -= timestep

        last_update = current_time
        viewer.render()
    else:
        break

# close
viewer.close()
