"""Client loop for KOS."""

import asyncio

import pykos

from kos_sim import logger


async def main() -> None:
    kos = pykos.KOS(ip="localhost", port=50051)
    sim = kos.simulation

    sim.set_parameters(time_scale=1.0)


if __name__ == "__main__":
    asyncio.run(main())
