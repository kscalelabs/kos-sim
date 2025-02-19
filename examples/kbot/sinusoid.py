"""Adhoc testing script."""

import asyncio
import math
import time

from pykos import KOS


async def main() -> None:
    async with KOS() as kos:
        while True:
            # Sinusoid.
            t = time.time()
            amplitude = math.sin(t * 2 * math.pi * 0.1) * 10.0
            pos = math.sin(t * 2 * math.pi) * amplitude - 60.0
            vel = math.cos(t * 2 * math.pi) * amplitude * 2 * math.pi

            await kos.actuator.command_actuators(
                [
                    {
                        "actuator_id": 14,
                        "position": pos,
                        "velocity": vel,
                    },
                    {
                        "actuator_id": 24,
                        "position": -pos,
                        "velocity": -vel,
                    },
                ]
            )

            await asyncio.sleep(0.02)


if __name__ == "__main__":
    # python -m examples.kbot.test
    asyncio.run(main())
