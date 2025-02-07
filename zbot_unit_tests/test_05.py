"""Tests running ZMP-based walking on the real robot."""

import asyncio
import logging

import colorlogging

logger = logging.getLogger(__name__)


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting test-05")
    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())
