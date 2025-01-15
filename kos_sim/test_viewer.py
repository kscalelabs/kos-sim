"""Test script for the viewer."""

import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path("models/gpr/robot_fixed.xml")
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(100000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
