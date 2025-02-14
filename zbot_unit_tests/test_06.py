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
from typing import Optional, Protocol

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
    """Results from running the IMU performance test."""

    avg_calls_per_second: float
    total_calls: int
    duration: float
    timestamps: list[float]
    samples_per_second: list[int]
    imu_readings: list[ImuValues]


# =============================================================================
# PERFORMANCE TEST FUNCTIONS
# =============================================================================


async def run_imu_test(kos: pykos.KOS, duration_seconds: int = 5) -> ImuTestResults:
    """Run the IMU performance test for the specified duration."""
    count = 0
    start_time = time.time()
    end_time = start_time + duration_seconds

    timestamps: list[float] = []
    samples_per_second: list[int] = []
    imu_readings: list[ImuValues] = []

    last_second = int(start_time)
    second_count = 0

    while time.time() < end_time:
        imu_values = await kos.imu.get_imu_values()
        count += 1
        second_count += 1
        imu_readings.append(imu_values)

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
    """Plot the IMU test results."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_rate, ax_accel, ax_gyro, ax_mag = axs.flatten()

    times = np.linspace(0, results.duration, len(results.imu_readings))

    # Plot 1: Sampling rate over time.
    ax_rate.plot(
        results.timestamps[1:], results.samples_per_second[1:], marker="o", linestyle="-", label="Samples/second"
    )
    ax_rate.set_xlabel("Time (seconds)")
    ax_rate.set_ylabel("Samples per Second")
    ax_rate.set_title("IMU Sampling Rate Over Time")
    ax_rate.grid(True)
    ax_rate.legend()

    # Plot 2: IMU Acceleration values over time.
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

    # Plot 3: IMU Gyroscope values over time.
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

    # Plot 4: IMU Magnetometer values over time.
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


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (in radians) to a rotation matrix (R = Rz @ Ry @ Rx)."""
    r_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    r_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    r_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return r_z @ r_y @ r_x


def reset_3d_axis(
    ax: Axes3D,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
) -> None:
    """Reset a 3D axis with the provided limits, labels, and title."""
    ax.cla()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)


async def realtime_orientation_plot(kos: pykos.KOS, duration_seconds: int = 10) -> None:
    """Display a real-time 3D plot of the IMU orientation, linear acceleration, and gravity vector.

    This function creates a single figure with three 3D subplots:
      1. Orientation (coordinate frame based on fused Euler angles).
      2. Linear acceleration vector.
      3. Gravity vector.

    Each subplot shows numerical annotations (values on separate lines).

    Args:
        kos: KOS client instance.
        duration_seconds: Duration to display the realtime plot.
    """
    plt.ion()  # Enable interactive mode.
    fig, (ax_orient, ax_accel, ax_grav) = plt.subplots(1, 3, figsize=(18, 8), subplot_kw={"projection": "3d"})

    # Initialize subplots.
    reset_3d_axis(ax_orient, (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), "X", "Y", "Z", "Real-time IMU Orientation")
    reset_3d_axis(ax_accel, (-10, 10), (-10, 10), (-10, 10), "X", "Y", "Z", "Real-time Acceleration")
    reset_3d_axis(ax_grav, (-10, 10), (-10, 10), (-10, 10), "X", "Y", "Z", "Real-time Gravity")

    start_time = time.time()
    last_second = int(start_time)
    second_count = 0

    while time.time() < start_time + duration_seconds:
        # Retrieve IMU data concurrently.
        imu_euler, imu_quat, imu_values, imu_advanced_values = await asyncio.gather(
            kos.imu.get_euler_angles(),
            kos.imu.get_quaternion(),
            kos.imu.get_imu_values(),
            kos.imu.get_imu_advanced_values(),
        )

        second_count += 1
        current_time = time.time()
        current_second = int(current_time)
        if current_second != last_second:
            logger.info(
                "Realtime Plot - Time: %.2f seconds - Calls this second: %d", current_time - start_time, second_count
            )
            second_count = 0
            last_second = current_second

        # Process Euler angles.
        roll_deg = imu_euler.roll
        pitch_deg = imu_euler.pitch
        yaw_deg = imu_euler.yaw

        # Convert angles to radians for computing the rotation matrix.
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        r = euler_to_rotation_matrix(roll, pitch, yaw)

        # Calculate rotated coordinate axes.
        x_axis = r @ np.array([1, 0, 0])
        y_axis = r @ np.array([0, 1, 0])
        z_axis = r @ np.array([0, 0, 1])

        # Reset subplots for the next frame.
        reset_3d_axis(ax_orient, (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), "X", "Y", "Z", "Real-time IMU Orientation")
        reset_3d_axis(ax_accel, (-10, 10), (-10, 10), (-10, 10), "X", "Y", "Z", "Real-time Acceleration")
        reset_3d_axis(ax_grav, (-10, 10), (-10, 10), (-10, 10), "X", "Y", "Z", "Real-time Gravity")

        # Orientation subplot: plot coordinate axes and annotate Euler angles.
        ax_orient.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color="r", label="X")
        ax_orient.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color="g", label="Y")
        ax_orient.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color="b", label="Z")
        orientation_text = f"Euler Angles:\nroll: {roll_deg:.2f}°\npitch: {pitch_deg:.2f}°\nyaw: {yaw_deg:.2f}°"
        ax_orient.text2D(
            0.05, -0.4, orientation_text, transform=ax_orient.transAxes, color="black", fontsize=24, clip_on=False
        )

        # Acceleration subplot: plot linear acceleration and annotate.
        accel_vec = np.array([imu_values.accel_x, imu_values.accel_y, imu_values.accel_z])
        ax_accel.quiver(0, 0, 0, accel_vec[0], accel_vec[1], accel_vec[2], color="m", label="Accel")
        accel_text = f"Linear Acceleration:\nx: {accel_vec[0]:.2f}\ny: {accel_vec[1]:.2f}\nz: {accel_vec[2]:.2f}"
        ax_accel.text2D(0.05, -0.4, accel_text, transform=ax_accel.transAxes, color="black", fontsize=24, clip_on=False)

        # Gravity subplot: plot gravity vector and annotate.
        gravity_vec = np.array([imu_advanced_values.grav_x, imu_advanced_values.grav_y, imu_advanced_values.grav_z])
        ax_grav.quiver(0, 0, 0, gravity_vec[0], gravity_vec[1], gravity_vec[2], color="c", label="Gravity")
        gravity_text = f"Gravity Vector:\nx: {gravity_vec[0]:.2f}\ny: {gravity_vec[1]:.2f}\nz: {gravity_vec[2]:.2f}"
        ax_grav.text2D(0.05, -0.4, gravity_text, transform=ax_grav.transAxes, color="black", fontsize=24, clip_on=False)

        plt.draw()
        plt.pause(0.001)

    plt.ioff()
    plt.show()


async def main() -> None:
    """Main entry point for the IMU test.

    Run either the realtime 3D orientation plot or the performance test (with plots).
    """
    parser = argparse.ArgumentParser(description="IMU Test Runner")
    parser.add_argument("--realtime_plot", action="store_true", help="Run realtime 3D orientation plot.")
    args = parser.parse_args()
    realtime_plot = args.realtime_plot

    colorlogging.configure()
    logger.warning("Starting IMU Test (test_06)")

    try:
        async with pykos.KOS("192.168.42.1") as kos:
            if realtime_plot:
                await realtime_orientation_plot(kos, duration_seconds=10000)
            else:
                results = await run_imu_test(kos, duration_seconds=5)
                plot_results(results)
    except Exception:
        logger.exception("Test failed. Ensure Z-Bot is connected via USB and the IP is accessible.")
        raise


if __name__ == "__main__":
    asyncio.run(main())
