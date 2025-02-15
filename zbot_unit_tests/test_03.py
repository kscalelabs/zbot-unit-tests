import asyncio
import sys
import subprocess
import time
from pykos import KOS

ACTUATOR_MAPPING = {
    "left_shoulder_yaw": 11,
    "left_shoulder_pitch": 12,
    "left_elbow": 13,
    "left_gripper": 14,
    "right_shoulder_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_elbow": 23,
    "right_gripper": 24,
    "left_hip_roll": 31,
    "left_hip_yaw": 32,
    "left_hip_pitch": 33,
    "left_knee": 34,
    "left_ankle": 35,
    "right_hip_roll": 41,
    "right_hip_yaw": 42,
    "right_hip_pitch": 43,
    "right_knee": 44,
    "right_ankle": 45,
}


async def run_robot(kos: KOS, is_real: bool) -> None:

    for actuator_id in ACTUATOR_MAPPING.values():
        if is_real:
            kp, kd = 32, 32
        else:
            gains = {actuator_id: (50, 15) for actuator_id in ACTUATOR_MAPPING.values()}
            kp, kd = gains.get(actuator_id, (32, 32))

        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=kp,
            kd=kd,
            max_torque=100,
            torque_enabled=True,
        )
    return kos


async def main():

    print("Starting simulator server...")
    sim_process = subprocess.Popen(["kos-sim", "zbot-v2-fixed", "--no-gravity"])
    time.sleep(2)

    try:
        print("Running on both simulator and real robot simultaneously...")
        async with KOS(ip="localhost", port=50051) as sim_kos, KOS(ip="10.33.11.170", port=50051) as real_kos:

            await sim_kos.sim.reset()

            sim_kos = await run_robot(sim_kos, False)
            real_kos = await run_robot(real_kos, True)

            print("Zeroing all joints...")
            zero_commands = [{"actuator_id": actuator_id, "position": 0} for actuator_id in ACTUATOR_MAPPING.values()]
            await asyncio.gather(
                sim_kos.actuator.command_actuators(zero_commands), real_kos.actuator.command_actuators(zero_commands)
            )
            await asyncio.sleep(2)

            for joint_name, actuator_id in ACTUATOR_MAPPING.items():
                print(f"Testing {joint_name} (ID: {actuator_id})")
                test_angle = -45

                command = [
                    {"actuator_id": actuator_id, "position": -test_angle if actuator_id in [21, 23, 24] else test_angle}
                ]
                sim_command = [
                    {"actuator_id": actuator_id, "position": -test_angle if actuator_id in [21, 23, 24] else test_angle}
                ]
                await asyncio.gather(
                    sim_kos.actuator.command_actuators(sim_command), real_kos.actuator.command_actuators(command)
                )
                await asyncio.sleep(2)

                command = [{"actuator_id": actuator_id, "position": 0}]
                sim_command = [{"actuator_id": actuator_id, "position": 0}]
                await asyncio.gather(
                    sim_kos.actuator.command_actuators(sim_command), real_kos.actuator.command_actuators(command)
                )
                await asyncio.sleep(1)

    finally:
        sim_process.terminate()
        sim_process.wait()


if __name__ == "__main__":
    asyncio.run(main())
