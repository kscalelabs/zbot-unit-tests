import argparse
import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass

import colorlogging
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(11, 1, 85.0, 2.0, 60.0, "left_shoulder_pitch_03"),
    Actuator(12, 5, 85.0, 2.0, 60.0, "left_shoulder_roll_03"),
    Actuator(13, 9, 30.0, 1.0, 17.0, "left_shoulder_yaw_02"),
    Actuator(14, 13, 30.0, 1.0, 17.0, "left_elbow_02"),
    Actuator(15, 17, 30.0, 1.0, 17.0, "left_wrist_02"),

    Actuator(21, 3, 85.0, 2.0, 60.0, "right_shoulder_pitch_03"),
    Actuator(22, 7, 85.0, 2.0, 60.0, "right_shoulder_roll_03"),
    Actuator(23, 11, 30.0, 1.0, 17.0, "right_shoulder_yaw_02"),
    Actuator(24, 15, 30.0, 1.0, 17.0, "right_elbow_02"),
    Actuator(25, 19, 30.0, 1.0, 17.0, "right_wrist_02"),

    Actuator(31, 0, 85.0, 3.0, 80.0, "left_hip_pitch_04"),
    Actuator(32, 4, 85.0, 2.0, 60.0, "left_hip_roll_03"),
    Actuator(33, 8, 30.0, 2.0, 60.0, "left_hip_yaw_03"),
    Actuator(34, 12, 60.0, 2.0, 80.0, "left_knee_04"),
    Actuator(35, 16, 80.0, 1.0, 17.0, "left_ankle_02"),

    Actuator(41, 2, 85.0, 3.0, 80.0, "right_hip_pitch_04"),
    Actuator(42, 6, 85.0, 2.0, 60.0, "right_hip_roll_03"),
    Actuator(43, 10, 30.0, 2.0, 60.0, "right_hip_yaw_03"),
    Actuator(44, 14, 60.0, 2.0, 80.0, "right_knee_04"),
    Actuator(45, 18, 80.0, 1.0, 17.0, "right_ankle_02"),  
]

async def configure_robot(kos: KOS) -> None:
    for actuator in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id = actuator.actuator_id,
            kp = actuator.kp,
            kd = actuator.kd,
            max_torque = actuator.max_torque,
            torque_enabled = True
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sim-only", action="store_true",
                        help="Run simulation only without connecting to the real robot")
    args = parser.parse_args()
 
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    sim_process = subprocess.Popen(["kos-sim", "kbot-v1", "--no-gravity"])
    time.sleep(2)
    try:
        if args.sim_only:
            print("Running in simulation-only mode...")
            async with KOS(ip=args.host, port=args.port) as sim_kos:
                await sim_kos.sim.reset()
                await configure_robot(sim_kos)
                print("Homing...")
                homing_command = [{
                    "actuator_id": actuator.actuator_id,
                    "position": 0.0
                } for actuator in ACTUATOR_LIST]
                
                await sim_kos.actuator.command_actuators(homing_command)
                await asyncio.sleep(3)

                order = [11, 21, 12, 22, 13, 23, 14, 24, 15, 25, 31, 41, 32, 42, 33, 43, 34, 44, 35, 45]


                actuator_map = {actuator.actuator_id: actuator for actuator in ACTUATOR_LIST}
                
                for actuator_id in order:
                    actuator = actuator_map.get(actuator_id)
                    print(f"Testing {actuator.actuator_id} in simulation...")

                    TEST_ANGLE = -10.0

                    FLIP_SIGN = 1.0 if actuator.actuator_id in [12, 21, 13, 14, 15, 25, 32, 33, 35, 41, 44] else -1.0
                    TEST_ANGLE *= FLIP_SIGN

                    command = [{
                        "actuator_id": actuator.actuator_id,
                        "position": TEST_ANGLE,
                    }]
                    print(f"Sending command to actuator {actuator.actuator_id}: position {TEST_ANGLE}")

                    await sim_kos.actuator.command_actuators(command)
                    await asyncio.sleep(2)

                    command = [{
                        "actuator_id": actuator.actuator_id,
                        "position": 0.0,
                    }]
                    await sim_kos.actuator.command_actuators(command)
                    await asyncio.sleep(2)
        else:
            print("Running on both simulator and real robots simultaneously...")
            async with KOS(ip=args.host, port=args.port) as sim_kos, \
                       KOS(ip="100.89.14.31", port=args.port) as real_kos:
                await sim_kos.sim.reset()
                await configure_robot(sim_kos)
                await configure_robot(real_kos)
                print("Homing...")
                homing_command = [{
                    "actuator_id": actuator.actuator_id,
                    "position": 0.0
                } for actuator in ACTUATOR_LIST]
                await asyncio.gather(
                    sim_kos.actuator.command_actuators(homing_command),
                    real_kos.actuator.command_actuators(homing_command),
                )
                await asyncio.sleep(2)


                order = [11, 21, 12, 22, 13, 23, 14, 24, 15, 25, 31, 41, 32, 42, 33, 43, 34, 44, 35, 45]

                actuator_map = {actuator.actuator_id: actuator for actuator in ACTUATOR_LIST}

                for actuator_id in order:
                        actuator = actuator_map.get(actuator_id)
                        print(f"Testing {actuator.actuator_id} in simulation...")

                        TEST_ANGLE = -10.0

                        FLIP_SIGN = 1.0 if actuator.actuator_id in [12, 21, 13, 14, 15, 25, 32, 33, 35, 41, 44] else -1.0
                        TEST_ANGLE *= FLIP_SIGN

                        command = [{
                            "actuator_id": actuator.actuator_id,
                            "position": TEST_ANGLE,
                        }]
                        print(f"Sending command to actuator {actuator.actuator_id}: position {TEST_ANGLE}")

                        await asyncio.gather(
                            sim_kos.actuator.command_actuators(command),
                            real_kos.actuator.command_actuators(command),
                        )
                        await asyncio.sleep(2)

                        command = [{
                            "actuator_id": actuator.actuator_id,
                            "position": 0.0,
                        }]
                        await asyncio.gather(
                            sim_kos.actuator.command_actuators(command),
                            real_kos.actuator.command_actuators(command),
                        )
                        
                        await asyncio.sleep(2)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        sim_process.terminate()
        sim_process.wait()


if __name__ == "__main__":
    asyncio.run(main())