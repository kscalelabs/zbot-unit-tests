"""Runs basic performance profiling using the KOS SDK."""

import asyncio
import logging
import time

import colorlogging
import matplotlib.pyplot as plt
import pykos

logger = logging.getLogger(__name__)


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting test-00")
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await performance_test(kos, 30)
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise


# Function to measure number of calls per second
async def performance_test(kos: pykos.KOS, duration_seconds: int = 10) -> None:
    count = 0
    start_time = time.time()
    end_time = start_time + duration_seconds
    timestamps = []
    counts_per_second = []

    last_second = int(time.time())
    second_count = 0

    while time.time() < end_time:
        await asyncio.gather(
            kos.actuator.get_actuators_state([11, 12, 13, 14]),
            kos.imu.get_imu_values(),
            kos.actuator.command_actuators([]),
        )
        count += 3
        second_count += 3

        current_second = int(time.time())
        if current_second != last_second:
            timestamps.append(current_second - start_time)
            counts_per_second.append(second_count)
            logger.info("Time: %.2f seconds - Calls this second: %d", current_second - start_time, second_count)
            second_count = 0
            last_second = current_second

    elapsed_time = time.time() - start_time
    logger.info("Total calls: %d", count)
    logger.info("Elapsed time: %.2f seconds", elapsed_time)
    logger.info("Calls per second: %.2f", count / elapsed_time)

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(timestamps[1:], counts_per_second[1:], marker="o", linestyle="-")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Calls per second")
    plt.title("Performance Test: Calls per Second Over Time")
    plt.grid()
    plt.show()


# Run the performance test for 10 seconds
if __name__ == "__main__":
    asyncio.run(main())
