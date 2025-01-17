"""Test script for the simulator."""

import argparse
import time

from kos_sim.simulator import MujocoSimulator


def test_simulation(model_path: str, duration: float = 5.0, speed: float = 1.0, render: bool = True) -> None:
    simulator = MujocoSimulator(model_path, render=render)

    for _ in range(int(duration / 0.001)):
        simulator.step()
        time.sleep(0.001)

    simulator.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the MuJoCo simulation.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to MuJoCo XML model file")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration to run simulation (seconds)")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()
    test_simulation(args.model_path, duration=args.duration, speed=args.speed, render=not args.no_render)


if __name__ == "__main__":
    # python -m kos_sim.test_simulator
    main()
