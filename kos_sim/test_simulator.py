"""Test script for the simulator."""

import argparse
import asyncio

from kscale import K

from kos_sim.simulator import MujocoSimulator


async def test_simulation(model_name: str, duration: float = 5.0, speed: float = 1.0, render: bool = True) -> None:
    api = K()
    bot_dir = await api.download_and_extract_urdf(model_name)
    bot_mjcf = next(bot_dir.glob("*.mjcf"))

    simulator = MujocoSimulator(bot_mjcf, render=render)

    timestep = simulator.timestep
    initial_update = last_update = asyncio.get_event_loop().time()

    while True:
        current_time = asyncio.get_event_loop().time()
        if current_time - initial_update > duration:
            break

        sim_time = current_time - last_update
        last_update = current_time
        while sim_time > 0:
            simulator.step()
            sim_time -= timestep

        simulator.render()

    simulator.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test the MuJoCo simulation.")
    parser.add_argument("model_name", type=str, help="Name of the model to simulate")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration to run simulation (seconds)")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()
    await test_simulation(args.model_name, duration=args.duration, speed=args.speed, render=not args.no_render)


if __name__ == "__main__":
    # python -m kos_sim.test_simulator
    asyncio.run(main())
