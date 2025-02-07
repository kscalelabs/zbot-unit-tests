"""Runs movement tests using the KOS SDK."""

import asyncio
import logging
import math
import time

import colorlogging
import matplotlib.pyplot as plt
import pykos

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


LEFT_ARM_ACTUATORS = [11, 12, 13, 14]
RIGHT_ARM_ACTUATORS = [21, 22, 23, 24]
LEFT_LEG_ACTUATORS = [31, 32, 33, 34, 35]
RIGHT_LEG_ACTUATORS = [41, 42, 43, 44, 45]

ALL_ACTUATORS = LEFT_ARM_ACTUATORS + RIGHT_ARM_ACTUATORS + LEFT_LEG_ACTUATORS + RIGHT_LEG_ACTUATORS


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting test-01")
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await movement_test(kos)
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise


async def movement_test(kos: pykos.KOS) -> None:
    """Runs sinusoidal movements on each of the K-Bot's limbs, one at a time."""
    all_data = []
    limbs = [LEFT_ARM_ACTUATORS, RIGHT_ARM_ACTUATORS, LEFT_LEG_ACTUATORS, RIGHT_LEG_ACTUATORS]

    for limb in limbs:
        timestamps, counts = await move_limb(kos, limb)
        all_data.append((limb, timestamps, counts))

    # Create a 2x2 grid of subplots
    _, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot data for each limb
    for idx, (limb, timestamps, counts) in enumerate(all_data):
        ax = axes[idx]
        ax.plot(timestamps[1:], counts[1:], marker="o", linestyle="-")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Instructions per second")
        ax.set_title(f"Performance Test: Limb {limb}")
        ax.grid()

    plt.tight_layout()
    plt.show()


async def move_limb(
    kos: pykos.KOS,
    limb: list[int],
    amplitude: float = 15.0,
    frequency: float = 0.1,
    repetitions: int = 500,
) -> tuple[list[float], list[int]]:
    """Moves a single limb of the K-Bot and returns performance data."""
    for actuator_id in limb:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=32.0,
            kd=32.0,
            torque_enabled=True,
        )

    # Reads start positions.
    states = await kos.actuator.get_actuators_state(limb)
    start_positions = {state.actuator_id: state.position for state in states.states}
    missing_ids = set(limb) - set(start_positions.keys())
    if missing_ids:
        raise ValueError(f"Actuator IDs {missing_ids} not found in start positions")

    start_time = time.time()

    # Move limbs and track instructions per second
    count = 0
    timestamps = []
    counts_per_second = []
    last_second = int(time.time())
    second_count = 0

    for _ in range(repetitions):
        current_time = time.time()
        elapsed_seconds = current_time - start_time
        target_delta = math.sin(elapsed_seconds * frequency * 2 * math.pi) * amplitude

        # Create commands for each actuator
        commands = [
            {
                "actuator_id": i,
                "position": start_positions[i] + target_delta,
            }
            for i in limb
        ]

        # Log the movement commands
        logger.debug("t=%.4fs : delta=%.2f deg, Actuator commands:", elapsed_seconds, target_delta)
        for cmd in commands:
            logger.debug("     ID %d : %.2f deg", cmd["actuator_id"], cmd["position"])

        await kos.actuator.command_actuators(commands)
        count += 1
        second_count += 1

        current_second = int(time.time())
        if current_second != last_second:
            timestamps.append(current_second - start_time)
            counts_per_second.append(second_count)
            logger.info("Time: %.2f seconds - Instructions this second: %d", current_second - start_time, second_count)
            second_count = 0
            last_second = current_second

    elapsed_time = time.time() - start_time
    logger.info("Total instructions: %d", count)
    logger.info("Elapsed time: %.2f seconds", elapsed_time)
    logger.info("Instructions per second: %.2f", count / elapsed_time)

    return timestamps, counts_per_second


if __name__ == "__main__":
    asyncio.run(main())
