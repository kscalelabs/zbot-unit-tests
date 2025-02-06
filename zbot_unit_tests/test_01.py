"""Runs movement tests using the KOS SDK."""

import asyncio
import logging
import math
import time
import colorlogging
import pykos

logger = logging.getLogger(__name__)

LEFT_ARM_ACTUATORS = [11, 12, 13, 14]
RIGHT_ARM_ACTUATORS = [21, 22, 23, 24]
LEFT_LEG_ACTUATORS = [31, 32, 33, 34, 35]
RIGHT_LEG_ACTUATORS = [41, 42, 43, 44, 45]

ALL_ACTUATORS = LEFT_ARM_ACTUATORS + RIGHT_ARM_ACTUATORS + LEFT_LEG_ACTUATORS + RIGHT_LEG_ACTUATORS


async def main() -> None:
    colorlogging.setup()
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await movement_test(kos)
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise


async def movement_test(kos: pykos.KOS) -> None:
    """Runs sinusoidal movements on each of the K-Bot's limbs, one at a time."""
    for limb in LEFT_ARM_ACTUATORS, RIGHT_ARM_ACTUATORS, LEFT_LEG_ACTUATORS, RIGHT_LEG_ACTUATORS:
        await move_limb(kos, limb)


async def move_limb(
    kos: pykos.KOS,
    limb: list[int],
    amplitude: float = 15.0,
    frequency: float = 1.0,
    repetitions: int = 5,
) -> None:
    """Moves a single limb of the K-Bot."""
    for actuator_id in limb:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=32.0,
            kd=32.0,
            torque_enabled=True,
        )

    # Reads start positions.
    states = await kos.actuator.get_actuators_state(limb)
    start_positions = [state.position for state in states.states]
    start_time = time.time()

    # Move limbs as fast as possible.
    for _ in range(repetitions):
        target_delta = math.sin((time.time() - start_time) * frequency * 2 * math.pi) * amplitude
        for i in range(len(limb)):
            await kos.actuator.configure_actuator(
                actuator_id=limb[i],
                position=start_positions[i] + target_delta,
            )


if __name__ == "__main__":
    asyncio.run(main())
