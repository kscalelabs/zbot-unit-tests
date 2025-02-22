"""Test to squat down and stand up and plot expected vs actual position."""

import argparse
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import colorlogging
import matplotlib.pyplot as plt
import numpy as np
from pykos import KOS

logger = logging.getLogger(__name__)


@dataclass
class StateLog:
    """Class for logging state data."""

    time: float
    actuator_id: int
    position: float
    velocity: float
    torque: float
    kp: float
    kd: float
    max_torque: float
    commanded_position: float
    commanded_velocity: float
    commanded_torque: float
    commanded_kp: float
    commanded_kd: float


@dataclass
class Actuator:
    """Class representing an actuator."""

    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


class DataLogger:
    """Class for logging and saving data."""

    def __init__(self):
        self.data: List[StateLog] = []

    def log_state(
        self,
        time: float,
        actuator_id: int,
        state,
        commanded_position: float,
        commanded_velocity: float,
        commanded_torque: float,
        commanded_kp: float,
        commanded_kd: float,
        actuator: Actuator,
    ) -> None:
        """Log a state entry."""
        self.data.append(
            StateLog(
                time=time,
                actuator_id=actuator_id,
                position=state.position,
                velocity=state.velocity,
                torque=state.torque,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque,
                commanded_position=commanded_position,
                commanded_velocity=commanded_velocity,
                commanded_torque=commanded_torque,
                commanded_kp=commanded_kp,
                commanded_kd=commanded_kd,
            )
        )

    def save_to_json(self, filename: str) -> None:
        """Save logged data to JSON file."""
        data = [
            {
                "time": d.time,
                "actuator_id": d.actuator_id,
                "position": d.position,
                "velocity": d.velocity,
                "torque": d.torque,
                "kp": d.kp,
                "kd": d.kd,
                "max_torque": d.max_torque,
                "commanded_position": d.commanded_position,
                "commanded_velocity": d.commanded_velocity,
                "commanded_torque": d.commanded_torque,
                "commanded_kp": d.commanded_kp,
                "commanded_kd": d.commanded_kd,
            }
            for d in self.data
        ]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


ACTUATOR_LIST = [
    # actuator id, nn id, kp, kd, max_torque, joint_name
    Actuator(11, 1, 150.0, 8.0, 60.0, "left_shoulder_pitch_03"),
    Actuator(12, 5, 150.0, 8.0, 60.0, "left_shoulder_roll_03"),
    Actuator(13, 9, 50.0, 5.0, 17.0, "left_shoulder_yaw_02"),
    Actuator(14, 13, 50.0, 5.0, 17.0, "left_elbow_02"),
    Actuator(15, 17, 20.0, 2.0, 17.0, "left_wrist_02"),
    Actuator(21, 3, 150.0, 8.0, 60.0, "right_shoulder_pitch_03"),
    Actuator(22, 7, 150.0, 8.0, 60.0, "right_shoulder_roll_03"),
    Actuator(23, 11, 50.0, 5.0, 17.0, "right_shoulder_yaw_02"),
    Actuator(24, 15, 50.0, 5.0, 17.0, "right_elbow_02"),
    Actuator(25, 19, 20.0, 2.0, 17.0, "right_wrist_02"),
    Actuator(31, 0, 300.0, 5.0, 100.0, "left_hip_pitch_04"),
    Actuator(32, 4, 300.0, 5.0, 100.0, "left_hip_roll_03"),
    Actuator(33, 8, 100.0, 5.0, 100.0, "left_hip_yaw_03"),
    Actuator(34, 12, 200.0, 5.0, 100.0, "left_knee_04"),
    Actuator(35, 16, 50.0, 1.0, 100.0, "left_ankle_02"),
    Actuator(41, 2, 300.0, 5.0, 100.0, "right_hip_pitch_04"),
    Actuator(42, 6, 300.0, 5.0, 100.0, "right_hip_roll_03"),
    Actuator(43, 10, 100.0, 5.0, 100.0, "right_hip_yaw_03"),
    Actuator(44, 14, 200.0, 5.0, 100.0, "right_knee_04"),
    Actuator(45, 18, 50.0, 1.0, 100.0, "right_ankle_02"),
]


async def smooth_move_joints(
    kos: KOS,
    actuator_ids: List[int],
    start_positions: List[float],
    target_positions: List[float],
    duration: float = 2.0,
) -> None:
    """
    Smoothly move joints from start positions to target positions over specified duration.

    Args:
        kos: KOS instance
        actuator_ids: List of actuator IDs
        start_positions: List of starting positions
        target_positions: List of target positions
        duration: Time in seconds for the movement
    """
    steps = int(duration * 50)  # 50Hz control frequency
    step_time = duration / steps

    for step in range(steps + 1):
        progress = step / steps
        # Use sinusoidal interpolation for smooth acceleration/deceleration
        smoothed_progress = (1 - math.cos(progress * math.pi)) / 2

        # Calculate intermediate positions
        current_positions = [
            start_pos + (target_pos - start_pos) * smoothed_progress
            for start_pos, target_pos in zip(start_positions, target_positions)
        ]

        # Prepare commands for all actuators
        commands = [{"actuator_id": act_id, "position": pos} for act_id, pos in zip(actuator_ids, current_positions)]

        try:
            await kos.actuator.command_actuators(commands)
        except Exception as e:
            logger.error(f"Failed to command actuators: {e}")
            raise

        await asyncio.sleep(step_time)


def plot_leg_data(sim_data: List[dict], leg: str = "left", real_data: Optional[List[dict]] = None) -> None:
    """
    Plot data for either left or right leg actuators with extended time range.

    Args:
        sim_data: List of simulation data points
        leg: Which leg to plot ("left" or "right")
        real_data: Optional list of real robot data points
    """
    try:
        # Filter actuator IDs for the specified leg
        leg_actuator_ranges = {
            "left": range(31, 36),  # Left leg actuator IDs
            "right": range(41, 46),  # Right leg actuator IDs
        }

        actuator_ids = sorted(
            list({d["actuator_id"] for d in sim_data if d["actuator_id"] in leg_actuator_ranges[leg]})
        )
        num_actuators = len(actuator_ids)

        # Calculate global time range with extended padding
        all_times = []
        for d in sim_data:
            all_times.append(d["time"])
        if real_data:
            for d in real_data:
                all_times.append(d["time"])

        time_min = min(all_times)
        time_max = max(all_times)

        # Extend time range by a fixed amount (e.g., 2 seconds) on each side
        # This ensures we see the full movement plus some padding
        time_padding = 2.0  # 2 seconds padding on each side
        global_xlim = (time_min - time_padding, time_max + time_padding)

        # Create figure with proper subplot grid - one row per actuator
        fig, axes = plt.subplots(num_actuators, 4, figsize=(20, 5 * num_actuators))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig.suptitle(f"{leg.title()} Leg Actuators Performance Analysis - {timestamp}", fontsize=16)

        # Handle case where there's only one actuator (axes won't be 2D)
        if num_actuators == 1:
            axes = [axes]

        # Plot each actuator
        for idx, actuator_id in enumerate(actuator_ids):
            actuator = next(a for a in ACTUATOR_LIST if a.actuator_id == actuator_id)
            logger.info(f"Plotting data for {actuator.joint_name} (ID: {actuator_id})")

            # Get data for this actuator
            sim_actuator_data = [d for d in sim_data if d["actuator_id"] == actuator_id]
            real_actuator_data = [d for d in real_data if d["actuator_id"] == actuator_id] if real_data else None

            if not sim_actuator_data:
                continue

            sim_times = [d["time"] for d in sim_actuator_data]
            real_times = [d["time"] for d in real_actuator_data] if real_actuator_data else None

            # Position plot
            ax1 = axes[idx][0]
            ax1.plot(sim_times, [d["commanded_position"] for d in sim_actuator_data], "b-", label="Sim Commanded")
            ax1.plot(sim_times, [d["position"] for d in sim_actuator_data], "b--", label="Sim Actual")
            if real_actuator_data:
                ax1.plot(
                    real_times, [d["commanded_position"] for d in real_actuator_data], "r-", label="Real Commanded"
                )
                ax1.plot(real_times, [d["position"] for d in real_actuator_data], "r--", label="Real Actual")
            ax1.set_title(f"{actuator.joint_name} Position")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Position (rad)")
            ax1.grid(True)
            ax1.legend()
            ax1.set_xlim(global_xlim)

            # Velocity plot
            ax2 = axes[idx][1]
            ax2.plot(sim_times, [d["velocity"] for d in sim_actuator_data], "b-", label="Sim Actual")
            ax2.plot(sim_times, [d["commanded_velocity"] for d in sim_actuator_data], "b--", label="Sim Commanded")
            if real_actuator_data:
                ax2.plot(real_times, [d["velocity"] for d in real_actuator_data], "r-", label="Real Actual")
                ax2.plot(
                    real_times, [d["commanded_velocity"] for d in real_actuator_data], "r--", label="Real Commanded"
                )
            ax2.set_title(f"{actuator.joint_name} Velocity")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Velocity (rad/s)")
            ax2.grid(True)
            ax2.legend()
            ax2.set_xlim(global_xlim)

            # Torque plot
            ax3 = axes[idx][2]
            ax3.plot(sim_times, [d["torque"] for d in sim_actuator_data], "b-", label="Sim Applied")
            ax3.plot(sim_times, [d["max_torque"] for d in sim_actuator_data], "b--", label="Sim Max")
            if real_actuator_data:
                ax3.plot(real_times, [d["torque"] for d in real_actuator_data], "r-", label="Real Applied")
                ax3.plot(real_times, [d["max_torque"] for d in real_actuator_data], "r--", label="Real Max")
            ax3.set_title(f"{actuator.joint_name} Torque")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Torque (Nm)")
            ax3.grid(True)
            ax3.legend()
            ax3.set_xlim(global_xlim)

            # Gains plot
            ax4 = axes[idx][3]
            ax4.plot(sim_times, [d["kp"] for d in sim_actuator_data], "b-", label="Sim Kp")
            ax4.plot(sim_times, [d["kd"] for d in sim_actuator_data], "b--", label="Sim Kd")
            if real_actuator_data:
                ax4.plot(real_times, [d["kp"] for d in real_actuator_data], "r-", label="Real Kp")
                ax4.plot(real_times, [d["kd"] for d in real_actuator_data], "r--", label="Real Kd")
            ax4.set_title(f"{actuator.joint_name} Gains")
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Gain Value")
            ax4.grid(True)
            ax4.legend()
            ax4.set_xlim(global_xlim)

            # Set major ticks every second
            for ax in [ax1, ax2, ax3, ax4]:
                ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))

        # Add timestamp to bottom of figure
        plt.figtext(0.99, 0.01, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ha="right", va="bottom", fontsize=8)

        # Save plot
        filename = f"squat_data_{leg}_leg_{timestamp}.png"
        plt.tight_layout()
        logger.info(f"Saving {leg} leg plot to {filename}")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Successfully created {leg} leg plot")

    except Exception as e:
        logger.error(f"Error creating {leg} leg plot: {e}")
        plt.close()


def plot_all_data(sim_data: List[dict], real_data: Optional[List[dict]] = None) -> None:
    """
    Create separate plots for left and right legs.
    """
    # Plot left leg
    plot_leg_data(sim_data, "left", real_data)

    # Plot right leg
    plot_leg_data(sim_data, "right", real_data)


async def configure_actuators(kos: KOS) -> None:
    """Configure actuators using values from test_00.py"""
    logger.info("Configuring actuators...")
    for actuator in ACTUATOR_LIST:
        try:
            adjusted_kp = actuator.kp * 0.9
            adjusted_kd = actuator.kd * 1.2

            logger.info(
                f"Configuring {actuator.joint_name} (ID: {actuator.actuator_id}) for smooth movement:\n"
                f"    kp: {adjusted_kp} (original: {actuator.kp})\n"
                f"    kd: {adjusted_kd} (original: {actuator.kd})\n"
                f"    max_torque: {actuator.max_torque}"
            )

            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=adjusted_kp,
                kd=adjusted_kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            )
        except Exception as e:
            logger.error(f"Failed to configure {actuator.joint_name}: {e}")
    logger.info("All actuators configured for smooth movement.")


async def squat_test(sim_kos: KOS, real_kos: KOS = None) -> None:
    """Test to squat down and stand up and plot expected vs actual position."""
    logger.info("Starting squat test...")

    # Initialize data loggers
    sim_logger = DataLogger()
    real_logger = DataLogger() if real_kos else None
    logger.info("Data loggers initialized")

    # Test parameters
    squat_depth = 45.0  # degrees
    duration = 4.0  # seconds for each movement (down and up)
    steps = 40  # number of interpolation steps per movement
    logger.info(f"Test parameters: squat_depth={squat_depth}°, duration={duration}s per movement, steps={steps}")

    # Define the joints involved in squatting
    squat_joints = {
        "hip_pitch": {"left": 31, "right": 41, "angle": -squat_depth},
        "knee": {"left": 34, "right": 44, "angle": -squat_depth * 1.5},  # Knees bend twice as much
        "ankle": {"left": 35, "right": 45, "angle": squat_depth * 0.8},  # Ankles compensate to keep balance
    }

    # Get all joint IDs involved in squatting
    all_ids = [joint["left"] for joint in squat_joints.values()] + [joint["right"] for joint in squat_joints.values()]
    logger.info(
        f"Configured joints for squatting: {[next(a.joint_name for a in ACTUATOR_LIST if a.actuator_id == id) for id in all_ids]}"
    )

    try:
        # Reset to initial position
        logger.info("Getting initial joint states...")
        start_states = await sim_kos.actuator.get_actuators_state(all_ids)
        start_positions = [state.position for state in start_states.states]
        logger.info(
            f"Initial positions: {dict(zip([next(a.joint_name for a in ACTUATOR_LIST if a.actuator_id == id) for id in all_ids], start_positions))}"
        )

        # Move to zero position first
        logger.info("Moving simulation to zero position...")
        await smooth_move_joints(sim_kos, all_ids, start_positions, [0.0] * len(all_ids), duration=2.0)
        if real_kos:
            try:
                logger.info("Moving real robot to zero position...")
                await smooth_move_joints(real_kos, all_ids, start_positions, [0.0] * len(all_ids), duration=2.0)
            except Exception as e:
                logger.error(f"Failed to move real robot: {e}")
                logger.warning("Disabling real robot control")
                real_kos = None

        logger.info("Waiting for stabilization...")
        await asyncio.sleep(1.0)

        # Phase 1: Squat Down
        logger.info("Starting squat down phase...")
        start_time = time.time()
        for step in range(steps + 1):
            current_time = time.time() - start_time
            progress = step / steps
            smoothed_progress = (1 - math.cos(progress * math.pi)) / 2

            if step % 5 == 0:
                logger.debug(f"Squat down - Step {step}/{steps} ({progress:.1%})")

            # Calculate positions for squatting down
            commands = []
            for joint_type, joint_info in squat_joints.items():
                target_angle = joint_info["angle"] * smoothed_progress
                left_angle = -target_angle  # Invert for left side
                right_angle = target_angle  # Keep same for right side

                commands.extend(
                    [
                        {"actuator_id": joint_info["left"], "position": left_angle, "velocity": 5.0},
                        {"actuator_id": joint_info["right"], "position": right_angle, "velocity": 5.0},
                    ]
                )
                logger.debug(f"{joint_type}: Left={left_angle:.2f}°, Right={right_angle:.2f}°")

            await execute_and_log_movement(sim_kos, real_kos, all_ids, commands, current_time, sim_logger, real_logger)
            await asyncio.sleep(duration / steps)

        # Hold at bottom position
        logger.info("Holding at bottom position...")
        await asyncio.sleep(1.0)

        # Phase 2: Stand Up
        logger.info("Starting stand up phase...")
        start_time = time.time()
        for step in range(steps + 1):
            current_time = time.time() - start_time + duration  # Offset time for continuous plotting
            progress = step / steps
            smoothed_progress = (1 - math.cos(progress * math.pi)) / 2

            if step % 5 == 0:
                logger.debug(f"Stand up - Step {step}/{steps} ({progress:.1%})")

            # Calculate positions for standing up (reverse of squat down)
            commands = []
            for joint_type, joint_info in squat_joints.items():
                target_angle = joint_info["angle"] * (1 - smoothed_progress)  # Reverse the movement
                left_angle = -target_angle  # Invert for left side
                right_angle = target_angle  # Keep same for right side

                commands.extend(
                    [
                        {"actuator_id": joint_info["left"], "position": left_angle, "velocity": 5.0},
                        {"actuator_id": joint_info["right"], "position": right_angle, "velocity": 5.0},
                    ]
                )
                logger.debug(f"{joint_type}: Left={left_angle:.2f}°, Right={right_angle:.2f}°")

            await execute_and_log_movement(sim_kos, real_kos, all_ids, commands, current_time, sim_logger, real_logger)
            await asyncio.sleep(duration / steps)

        # Save the logged data
        logger.info("Saving simulation data to squat_sim_data.json...")
        sim_logger.save_to_json("squat_sim_data.json")
        if real_logger and real_logger.data:
            logger.info("Saving real robot data to squat_real_data.json...")
            real_logger.save_to_json("squat_real_data.json")

    except Exception as e:
        logger.error(f"Error during squat test: {e}")
        raise
    finally:
        logger.info("Squat test complete!")


async def execute_and_log_movement(sim_kos, real_kos, all_ids, commands, current_time, sim_logger, real_logger):
    """Execute commands and log states for both simulation and real robot."""
    # Send commands and log states for simulation
    sim_states = await sim_kos.actuator.get_actuators_state(all_ids)
    await sim_kos.actuator.command_actuators(commands)

    # Log simulation data
    for i, actuator_id in enumerate(all_ids):
        actuator = next(a for a in ACTUATOR_LIST if a.actuator_id == actuator_id)
        sim_logger.log_state(
            time=current_time,
            actuator_id=actuator_id,
            state=sim_states.states[i],
            commanded_position=commands[i]["position"],
            commanded_velocity=5.0,  # Match the commanded velocity
            commanded_torque=0.0,
            commanded_kp=actuator.kp,
            commanded_kd=actuator.kd,
            actuator=actuator,
        )

    # Handle real robot if connected
    if real_kos:
        try:
            real_states = await real_kos.actuator.get_actuators_state(all_ids)
            await real_kos.actuator.command_actuators(commands)

            for i, actuator_id in enumerate(all_ids):
                actuator = next(a for a in ACTUATOR_LIST if a.actuator_id == actuator_id)
                real_logger.log_state(
                    time=current_time,
                    actuator_id=actuator_id,
                    state=real_states.states[i],
                    commanded_position=commands[i]["position"],
                    commanded_velocity=5.0,  # Match the commanded velocity
                    commanded_torque=0.0,
                    commanded_kp=actuator.kp,
                    commanded_kd=actuator.kd,
                    actuator=actuator,
                )
        except Exception as e:
            logger.error(f"Failed to communicate with real robot: {e}")
            logger.warning("Disabling real robot control")
            return None


async def main() -> None:
    """Main function to run the squat test."""
    parser = argparse.ArgumentParser(description="Squat test with plotting")
    parser.add_argument("--host", type=str, default="100.89.14.31", help="Simulation host")
    parser.add_argument("--port", type=int, default=50051, help="Port number")
    parser.add_argument("--real-host", type=str, default="100.89.14.31", help="Real robot host")
    parser.add_argument("--sim-only", action="store_true", help="Run only in simulation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    colorlogging.configure(level=log_level)
    logger.info(f"Starting squat test with configuration:")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Real Host: {args.real_host if not args.sim_only else 'disabled'}")
    logger.info(f"  Debug Mode: {'enabled' if args.debug else 'disabled'}")

    try:
        # Define squat joints for plotting
        squat_joints = {
            "shoulder_pitch": {"left": 11, "right": 21},
            "shoulder_roll": {"left": 12, "right": 22},
            "shoulder_yaw": {"left": 13, "right": 23},
            "elbow": {"left": 14, "right": 24},
            "wrist": {"left": 15, "right": 25},
            "hip_pitch": {"left": 31, "right": 41},
            "hip_roll": {"left": 32, "right": 42},
            "hip_yaw": {"left": 33, "right": 43},
            "knee": {"left": 34, "right": 44},
            "ankle": {"left": 35, "right": 45},
        }
        all_ids = [joint["left"] for joint in squat_joints.values()] + [
            joint["right"] for joint in squat_joints.values()
        ]
        logger.info(
            f"Configured joints for testing: {[next(a.joint_name for a in ACTUATOR_LIST if a.actuator_id == id) for id in all_ids]}"
        )

        if args.sim_only:
            logger.info("Running squat test in simulation only...")
            async with KOS(ip=args.host, port=args.port) as sim_kos:
                logger.info("Connected to simulator")
                logger.info("Resetting simulation state...")
                await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})
                await configure_actuators(sim_kos)
                await squat_test(sim_kos)

                # Load and plot simulation data
                logger.info("Loading simulation data for plotting...")
                with open("squat_sim_data.json", "r") as f:
                    sim_data = json.load(f)
                logger.info(f"Loaded {len(sim_data)} data points")

                logger.info("Generating consolidated plot...")
                plot_all_data(sim_data)
        else:
            logger.info("Running squat test on both simulator and real robot...")
            try:
                async with KOS(ip=args.host, port=args.port) as sim_kos:
                    logger.info("Connected to simulator")
                    logger.info("Attempting connection to real robot...")
                    async with KOS(ip=args.real_host, port=args.port) as real_kos:
                        logger.info("Connected to real robot")
                        logger.info("Resetting simulation state...")
                        # await sim_kos.sim.reset(
                        #     initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20}
                        # )

                        # Configure both robots
                        logger.info("Configuring simulator and real robot...")
                        await asyncio.gather(
                            configure_actuators(sim_kos),
                            configure_actuators(real_kos),
                        )

                        # Run squat test on both
                        await squat_test(sim_kos, real_kos)

                        # Load and plot both simulation and real data
                        logger.info("Loading data for plotting...")
                        with open("squat_sim_data.json", "r") as f:
                            sim_data = json.load(f)
                        logger.info(f"Loaded {len(sim_data)} simulation data points")

                        real_data = None
                        try:
                            with open("squat_real_data.json", "r") as f:
                                real_data = json.load(f)
                            logger.info(f"Loaded {len(real_data)} real robot data points")
                        except FileNotFoundError:
                            logger.warning("No real robot data found for plotting")

                        logger.info("Generating consolidated plot...")
                        plot_all_data(sim_data, real_data)

            except Exception as e:
                logger.error(f"Error during real robot connection/operation: {e}")
                logger.info("Continuing with simulation only...")
                # Fallback to simulation only
                async with KOS(ip=args.host, port=args.port) as sim_kos:
                    await sim_kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})
                    await configure_actuators(sim_kos)
                    await squat_test(sim_kos)

                    # Load and plot simulation data
                    logger.info("Loading simulation data for plotting...")
                    with open("squat_sim_data.json", "r") as f:
                        sim_data = json.load(f)
                    logger.info(f"Loaded {len(sim_data)} data points")

                    logger.info("Generating consolidated plot...")
                    plot_all_data(sim_data)

    except Exception as e:
        logger.error(f"Squat test failed with error: {e}")
        raise
    finally:
        logger.info("Squat test and plotting complete")


if __name__ == "__main__":
    asyncio.run(main())
