"""Runs reinforcement learning unit tests.

To see a video of the policy running on the robot, look in `assets/policy.mp4`.
"""

import asyncio
import logging

import colorlogging
import pykos

logger = logging.getLogger(__name__)


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting test-02")
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await reinforcement_learning_test(kos)
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise


async def reinforcement_learning_test(kos: pykos.KOS) -> None:
    """Runs reinforcement learning unit tests."""
    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())
