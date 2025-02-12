"""IMU Test Suite.

This test measures IMU performance and data quality. It performs the following:
1. Measures the maximum IMU sampling rate
2. Logs and visualizes IMU data stability
3. Reports key statistics about IMU performance

Use the --realtime_plot flag to display a real-time 3D plot of the IMU orientation.

"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import colorlogging
import matplotlib.pyplot as plt
import numpy as np
import pykos
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class ImuValues(Protocol):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: Optional[float]
    mag_y: Optional[float]
    mag_z: Optional[float]


@dataclass
class ImuTestResults:
    """Results from running the IMU test."""

    avg_calls_per_second: float
    total_calls: int
    duration: float
    timestamps: List[float]
    samples_per_second: List[int]
    imu_readings: List[ImuValues]


async def run_imu_test(kos: pykos.KOS, duration_seconds: int = 5) -> ImuTestResults:
    """Run IMU performance and data collection test.

    Args:
        kos: KOS client instance
        duration_seconds: How long to run the test for

    Returns:
        ImuTestResults containing test statistics and data
    """
    count = 0
    start_time = time.time()
    end_time = start_time + duration_seconds

    timestamps: List[float] = []
    samples_per_second: List[int] = []
    imu_readings: List[ImuValues] = []

    last_second = int(start_time)
    second_count = 0

    while time.time() < end_time:
        # Get IMU values
        imu_values = await kos.imu.get_imu_values()
        count += 1
        second_count += 1
        imu_readings.append(imu_values)

        # Track samples per second
        current_second = int(time.time())
        if current_second != last_second:
            timestamps.append(current_second - start_time)
            samples_per_second.append(second_count)
            logger.info("Time: %.2f seconds - Samples this second: %d", current_second - start_time, second_count)
            second_count = 0
            last_second = current_second

    elapsed_time = time.time() - start_time
    avg_rate = count / elapsed_time

    logger.info("Test Complete:")
    logger.info("Total samples: %d", count)
    logger.info("Elapsed time: %.2f seconds", elapsed_time)
    logger.info("Average sampling rate: %.2f Hz", avg_rate)

    return ImuTestResults(
        avg_calls_per_second=avg_rate,
        total_calls=count,
        duration=elapsed_time,
        timestamps=timestamps,
        samples_per_second=samples_per_second,
        imu_readings=imu_readings,
    )


def plot_results(results: ImuTestResults) -> None:
    """Generate plots visualizing the IMU test results.

    Args:
        results: The IMU test results to visualize
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_rate, ax_accel, ax_gyro, ax_mag = axs.flatten()

    times = np.linspace(0, results.duration, len(results.imu_readings))

    # Plot 1: Sampling rate over time
    ax_rate.plot(
        results.timestamps[1:], results.samples_per_second[1:], marker="o", linestyle="-", label="Samples/second"
    )
    ax_rate.set_xlabel("Time (seconds)")
    ax_rate.set_ylabel("Samples per Second")
    ax_rate.set_title("IMU Sampling Rate Over Time")
    ax_rate.grid(True)
    ax_rate.legend()

    # Plot 2: IMU acceleration values over time
    accel_x = [r.accel_x for r in results.imu_readings]
    accel_y = [r.accel_y for r in results.imu_readings]
    accel_z = [r.accel_z for r in results.imu_readings]
    ax_accel.plot(times, accel_x, label="Acc X")
    ax_accel.plot(times, accel_y, label="Acc Y")
    ax_accel.plot(times, accel_z, label="Acc Z")
    ax_accel.set_xlabel("Time (seconds)")
    ax_accel.set_ylabel("Acceleration (m/s²)")
    ax_accel.set_title("IMU Acceleration")
    ax_accel.grid(True)
    ax_accel.legend()

    # Plot 3: Gyroscope values over time
    gyro_x = [r.gyro_x for r in results.imu_readings]
    gyro_y = [r.gyro_y for r in results.imu_readings]
    gyro_z = [r.gyro_z for r in results.imu_readings]
    ax_gyro.plot(times, gyro_x, label="Gyro X")
    ax_gyro.plot(times, gyro_y, label="Gyro Y")
    ax_gyro.plot(times, gyro_z, label="Gyro Z")
    ax_gyro.set_xlabel("Time (seconds)")
    ax_gyro.set_ylabel("Gyro (deg/s)")
    ax_gyro.set_title("IMU Gyroscope")
    ax_gyro.grid(True)
    ax_gyro.legend()

    # Plot 4: Magnetometer values over time
    # Use 0.0 if magnetometer values are None
    mag_x = [r.mag_x if r.mag_x is not None else 0.0 for r in results.imu_readings]
    mag_y = [r.mag_y if r.mag_y is not None else 0.0 for r in results.imu_readings]
    mag_z = [r.mag_z if r.mag_z is not None else 0.0 for r in results.imu_readings]
    ax_mag.plot(times, mag_x, label="Mag X")
    ax_mag.plot(times, mag_y, label="Mag Y")
    ax_mag.plot(times, mag_z, label="Mag Z")
    ax_mag.set_xlabel("Time (seconds)")
    ax_mag.set_ylabel("Mag (units)")
    ax_mag.set_title("IMU Magnetometer")
    ax_mag.grid(True)
    ax_mag.legend()

    plt.tight_layout()
    plt.show()


def compute_euler_angles(imu_value: ImuValues) -> Tuple[float, float, float]:
    """Compute Euler angles (roll, pitch, yaw) from IMU sensor values.

    Roll and pitch are derived from the accelerometer and yaw is computed
    using a tilt compensation of the magnetometer.
    """
    # Compute roll and pitch from accelerometer
    roll = np.arctan2(imu_value.accel_y, imu_value.accel_z)
    pitch = np.arctan2(-imu_value.accel_x, np.sqrt(imu_value.accel_y**2 + imu_value.accel_z**2))

    # Tilt compensation for magnetometer (simple approach)
    mag_x = imu_value.mag_x * np.cos(pitch) + imu_value.mag_z * np.sin(pitch)
    mag_y = (
        imu_value.mag_x * np.sin(roll) * np.sin(pitch)
        + imu_value.mag_y * np.cos(roll)
        - imu_value.mag_z * np.sin(roll) * np.cos(pitch)
    )
    yaw = np.arctan2(-mag_y, mag_x)
    return roll, pitch, yaw


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to a rotation matrix.

    This rotation matrix is obtained using the sequence: R = Rz * Ry * Rx.
    """
    r_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    r_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    r_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return r_z.dot(r_y).dot(r_x)


def reset_3d_axis(
    ax: Axes3D,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
) -> None:
    """Reset a 3D axis with the given limits, labels, and title."""
    ax.cla()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)


def update_orientation(orientation: np.ndarray, gyro_x: float, gyro_y: float, gyro_z: float, dt: float) -> None:
    """Update the current Euler orientation (in radians) by integrating the gyro angular velocities.

    The gyro values are given in degrees per second and are converted to radians per second.
    """
    orientation[0] += np.deg2rad(gyro_x) * dt
    orientation[1] += np.deg2rad(gyro_y) * dt
    orientation[2] += np.deg2rad(gyro_z) * dt


async def realtime_orientation_plot(kos: pykos.KOS, duration_seconds: int = 10) -> None:
    """Display a real-time 3D plot of the IMU orientation.

    The plot uses quiver arrows to denote the rotated coordinate frame.
    """
    plt.ion()  # enable interactive mode
    fig, (ax_orient, ax_accel) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

    # Set axis properties for orientation subplot
    reset_3d_axis(ax_orient, (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), "X", "Y", "Z", "Real-time IMU Orientation")
    # Set axis properties for acceleration subplot
    reset_3d_axis(ax_accel, (-10, 10), (-10, 10), (-10, 10), "X", "Y", "Z", "Real-time Acceleration Vector")

    start_time = time.time()
    last_update_time = start_time
    # Initialize the orientation as zero (upright, identity rotation)
    orientation_euler = np.array([0.0, 0.0, 0.0])
    last_second = int(start_time)
    second_count = 0

    while time.time() < start_time + duration_seconds:
        imu_value = await kos.imu.get_imu_values()
        second_count += 1

        current_time = time.time()
        dt = current_time - last_update_time
        last_update_time = current_time
        # Update the orientation by integrating gyro angular velocities (deg/s)
        update_orientation(orientation_euler, imu_value.gyro_x, imu_value.gyro_y, imu_value.gyro_z, dt)

        current_second = int(time.time())
        if current_second != last_second:
            logger.info(
                "Realtime Plot - Time: %.2f seconds - Calls this second: %d",
                current_second - start_time,
                second_count,
            )
            second_count = 0
            last_second = current_second

        # Compute rotation matrix from the integrated Euler angles
        r = euler_to_rotation_matrix(orientation_euler[0], orientation_euler[1], orientation_euler[2])

        # Compute rotated coordinate axes (unit vectors) for orientation
        x_axis = r.dot(np.array([1, 0, 0]))
        y_axis = r.dot(np.array([0, 1, 0]))
        z_axis = r.dot(np.array([0, 0, 1]))

        # Clear and reset the axes in each iteration to remove old arrows
        reset_3d_axis(ax_orient, (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), "X", "Y", "Z", "Real-time IMU Orientation")
        reset_3d_axis(ax_accel, (-10, 10), (-10, 10), (-10, 10), "X", "Y", "Z", "Real-time Acceleration Vector")

        # Plot the coordinate frame on the orientation subplot
        ax_orient.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color="r", label="X")
        ax_orient.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color="g", label="Y")
        ax_orient.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color="b", label="Z")

        # Plot the acceleration vector on the acceleration subplot
        accel_vec = np.array([imu_value.accel_x, imu_value.accel_y, imu_value.accel_z])
        ax_accel.quiver(0, 0, 0, accel_vec[0], accel_vec[1], accel_vec[2], color="m", label="Accel")

        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()


async def main() -> None:
    """Main entry point for the IMU test.

    By default, the performance test and its plots are executed.
    If the --realtime_plot flag is provided, the real-time 3D orientation
    plot will be displayed afterward.
    """
    parser = argparse.ArgumentParser(description="IMU Test Runner")
    parser.add_argument(
        "--realtime_plot", action="store_true", help="Run 3D realtime orientation plot after the performance test."
    )
    args = parser.parse_args()
    realtime_plot = args.realtime_plot

    colorlogging.configure()
    logger.warning("Starting IMU Test (test-06)")

    try:
        async with pykos.KOS("192.168.42.1") as kos:
            if not realtime_plot:
                # Run the performance test (previous functionality)
                results = await run_imu_test(kos, duration_seconds=5)
                plot_results(results)
            else:
                # If the realtime_plot flag is set, display the 3D orientation plot
                await realtime_orientation_plot(kos, duration_seconds=10000)
    except Exception:
        logger.exception("Test failed. Ensure Z-Bot is connected via USB and the IP is accessible.")
        raise


if __name__ == "__main__":
    asyncio.run(main())
