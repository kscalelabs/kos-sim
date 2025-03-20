"""Example of saving a video from the simulation."""

import asyncio
import logging

from pykos import KOS

logger = logging.getLogger(__name__)

async def main() -> None:
    async with KOS() as kos:
        await kos.sim.reset(
            pos={"x": 0.0, "y": 0.0, "z": 2.0},
            quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        )
        clip = (await kos.process_manager.start_kclip(action="start")).clip_uuid
        await asyncio.sleep(5)
        await kos.process_manager.stop_kclip()
        logger.info("Clip saved to: %s", clip)


# This works only if you run the server with the `--no-render` flag
# `kos_sim default-humanoid --no-render`
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
