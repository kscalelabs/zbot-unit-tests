"""Interactive example script for a command to keep the robot balanced."""
#! MAKE SURE BOTH ARE OFF IN THE SAME WAY IN KOS SIM AND REAL

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
import json
from typing import Dict, List

import colorlogging
import numpy as np
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


@dataclass
class StateLog:
    time: float
    actuator_id: int
    position: float
    velocity: float
    commanded_position: float
    commanded_velocity: float


class TestData:
    def __init__(self):
        self.states: List[StateLog] = []
    
    def log_state(self, time: float, actuator_id: int, position: float, velocity: float, 
                 commanded_position: float, commanded_velocity: float):
        self.states.append(StateLog(time, actuator_id, position, velocity, 
                                  commanded_position, commanded_velocity))
    
    def save_to_json(self, filename: str):
        data = [{"time": s.time, "actuator_id": s.actuator_id, 
                 "position": s.position, "velocity": s.velocity,
                 "commanded_position": s.commanded_position,
                 "commanded_velocity": s.commanded_velocity}
                for s in self.states]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


ACTUATOR_LIST: list[Actuator] = [
    # actuator id, nn id, kp, kd, max_torque
    Actuator(11, 1, 150.0, 8.0, 60.0),  # left_shoulder_pitch_03
    Actuator(12, 5, 150.0, 8.0, 60.0),  # left_shoulder_roll_03
    Actuator(13, 9, 50.0, 5.0, 17.0),  # left_shoulder_yaw_02
    Actuator(14, 13, 50.0, 5.0, 17.0),  # left_elbow_02
    Actuator(15, 17, 20.0, 2.0, 17.0),  # left_wrist_02
    Actuator(21, 3, 150.0, 8.0, 60.0),  # right_shoulder_pitch_03
    Actuator(22, 7, 150.0, 8.0, 60.0),  # right_shoulder_roll_03
    Actuator(23, 11, 50.0, 5.0, 17.0),  # right_shoulder_yaw_02
    Actuator(24, 15, 50.0, 5.0, 17.0),  # right_elbow_02
    Actuator(25, 19, 20.0, 2.0, 17.0),  # right_wrist_02
    Actuator(31, 0, 85.0, 3.0, 80.0),  # left_hip_pitch_04 (RS04_Pitch)
    Actuator(32, 4, 85.0, 2.0, 60.0),  # left_hip_roll_03 (RS03_Roll)
    Actuator(33, 8, 30.0, 2.0, 60.0),  # left_hip_yaw_03 (RS03_Yaw)
    Actuator(34, 12, 60.0, 2.0, 80.0),  # left_knee_04 (RS04_Knee)
    Actuator(35, 16, 80.0, 1.0, 17.0),  # left_ankle_02 (RS02)
    Actuator(41, 2, 85.0, 3.0, 80.0),  # right_hip_pitch_04 (RS04_Pitch)
    Actuator(42, 6, 85.0, 2.0, 60.0),  # right_hip_roll_03 (RS03_Roll)
    Actuator(43, 10, 30.0, 2.0, 60.0),  # right_hip_yaw_03 (RS03_Yaw)
    Actuator(44, 14, 100.0, 7.0, 80.0),  # right_knee_04 (RS04_Knee)
    Actuator(45, 18, 80.0, 1.0, 17.0),  # right_ankle_02 (RS02)
]

# Define the actuators we want to move
ACTUATORS_TO_MOVE = [44]  # left/right hip roll and left/right ankle

async def test_client(sim_kos: KOS, real_kos: KOS = None) -> None:
    logger.info("Starting test client...")
    
    # Initialize separate test data loggers for sim and real
    sim_data = TestData()
    real_data = TestData() if real_kos else None

    # Test parameters
    amplitude = 10.0   # degrees (half of peak-to-peak)
    offset = -25.0  # degrees (center point between -10 and -20)
    frequency = 4  # Hz
    duration = 10.0   # seconds
    
    # Reset the simulation
    await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})

    # First disable all motors
    logger.info("Disabling motors...")
    for actuator in ACTUATOR_LIST:
        await sim_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
        if real_kos:
            await real_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
    
    await asyncio.sleep(1)

    # Configure motors with their gains
    logger.info("Configuring motors...")
    for actuator in ACTUATOR_LIST:
        config_commands = []
        config_commands.append(sim_kos.actuator.configure_actuator(
            actuator_id=actuator.actuator_id,
            torque_enabled=True,
        ))
        if real_kos:
            config_commands.append(real_kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            ))
        await asyncio.gather(*config_commands)

    start_time = time.time()
    next_time = start_time + 1 / 100  # 100Hz control rate

    while time.time() - start_time < duration:
        current_time = time.time()
        t = current_time - start_time
        
        # Calculate sine wave position with offset
        angular_freq = 2 * np.pi * frequency
        position = offset + amplitude * np.sin(angular_freq * t)
        # velocity = amplitude * angular_freq * np.cos(angular_freq * t)
        velocity = 0.0

        # Filter actuator list to only include actuators we want to move
        active_actuators = [actuator for actuator in ACTUATOR_LIST if actuator.actuator_id in ACTUATORS_TO_MOVE]

        commands = [
            {
                "actuator_id": actuator.actuator_id, 
                "position": position,
                "velocity": velocity
            }
            for actuator in active_actuators
        ]

        command_tasks = [sim_kos.actuator.command_actuators(commands)]
        state_tasks = [sim_kos.actuator.get_actuators_state(actuator.actuator_id for actuator in active_actuators)]
        
        if real_kos:
            command_tasks.append(real_kos.actuator.command_actuators(commands))
            state_tasks.append(real_kos.actuator.get_actuators_state(actuator.actuator_id for actuator in active_actuators))

        await asyncio.gather(*command_tasks)
        states_list = await asyncio.gather(*state_tasks)

        # Log states separately for sim and real
        for state in states_list[0].states:
            sim_data.log_state(t, state.actuator_id, state.position, state.velocity,
                             position, velocity)
        
        if real_kos and len(states_list) > 1:
            for state in states_list[1].states:
                real_data.log_state(t, state.actuator_id, state.position, state.velocity,
                                  position, velocity)

        if next_time > current_time:
            await asyncio.sleep(next_time - current_time)
        next_time += 1 / 100

    # Disable motors at the end
    logger.info("Disabling motors...")
    for actuator in ACTUATOR_LIST:
        disable_commands = [
            sim_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
        ]
        if real_kos:
            disable_commands.append(
                real_kos.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
            )
        await asyncio.gather(*disable_commands)

    # Save the logged data to separate JSON files
    sim_data.save_to_json("./utils/sim_actuator_states.json")
    if real_data:
        real_data.save_to_json("./utils/real_actuator_states.json")

async def main() -> None:
    """Runs the main simulation loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--deploy", action="store_true",
                       help="Connect to the real robot (default: simulation only)")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    if not args.deploy:
        async with KOS(ip=args.host, port=args.port) as sim_kos:
            await test_client(sim_kos)
    else:
        async with KOS(ip=args.host, port=args.port) as sim_kos, \
                   KOS(ip="100.89.14.31", port=args.port) as real_kos:
            await test_client(sim_kos, real_kos)


if __name__ == "__main__":
    # python -m examples.kbot.balancing
    asyncio.run(main())