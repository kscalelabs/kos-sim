import mujoco

model = mujoco.MjModel.from_xml_path("/Users/jchunx/.kscale/robots/kbot-v1/robot/k-bot_sim.mjcf")
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

print(data.qpos)