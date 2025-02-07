"""Runs Mujoco and real robot in parallel, for tuning the Z-Bot simulation parameters."""

import asyncio
import logging

import colorlogging
import pykos

logger = logging.getLogger(__name__)


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting test-04")
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await run_motor_system_identification(kos)
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise


async def run_motor_system_identification(kos: pykos.KOS) -> None:
    """Runs motor system identification."""
    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())
