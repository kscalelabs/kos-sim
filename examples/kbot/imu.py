"""Script to read the IMU data from the simulator."""

import argparse
import asyncio
import logging

import colorlogging
from pykos import KOS

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    async with KOS(ip=args.host, port=args.port) as kos:
        await kos.sim.reset(
            pos={"x": 0.0, "y": 0.0, "z": 1.25},
            quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        )

        while True:
            values = await kos.imu.get_imu_values()
            logger.info("Accel X: %.5f, Y: %.5f, Z: %.5f", values.accel_x, values.accel_y, values.accel_z)
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
