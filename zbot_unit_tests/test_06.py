"""IMU Test Suite.

This test measures IMU performance and data quality. It performs the following:
1. Measures the maximum IMU sampling rate
2. Logs and visualizes IMU data stability
3. Reports key statistics about IMU performance

"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple, Any

import colorlogging
import matplotlib.pyplot as plt
import numpy as np
import pykos
# from pykos.proto import ImuValues

logger = logging.getLogger(__name__)

@dataclass
class ImuTestResults:
    """Results from running the IMU test."""
    avg_calls_per_second: float
    total_calls: int
    duration: float
    timestamps: List[float]
    samples_per_second: List[int]
    imu_readings: List[Any]

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
    
    timestamps = []
    samples_per_second = []
    imu_readings = []

    last_second = int(start_time)
    second_count = 0

    while time.time() < end_time:
        # Get IMU values
        imu_values = await kos.imu.get_imu_values()
        count += 1
        second_count += 1
        imu_readings.append(imu_values)
        
        # # Log IMU values
        # logger.info(
        #     "IMU Values - Accel (m/s²): (%.2f, %.2f, %.2f) Gyro (rad/s): (%.2f, %.2f, %.2f) Mag: (%.2f, %.2f, %.2f)",
        #     imu_values.accel_x, imu_values.accel_y, imu_values.accel_z,
        #     imu_values.gyro_x, imu_values.gyro_y, imu_values.gyro_z,
        #     imu_values.mag_x or 0.0, imu_values.mag_y or 0.0, imu_values.mag_z or 0.0
        # )

        # Track samples per second
        current_second = int(time.time())
        if current_second != last_second:
            timestamps.append(current_second - start_time)
            samples_per_second.append(second_count)
            logger.info("Time: %.2f seconds - Samples this second: %d", 
                       current_second - start_time, second_count)
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
        imu_readings=imu_readings
    )

def plot_results(results: ImuTestResults) -> None:
    """Generate plots visualizing the IMU test results.
    
    Args:
        results: The IMU test results to visualize
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Sampling rate over time
    ax1.plot(results.timestamps[1:], results.samples_per_second[1:], 
             marker="o", linestyle="-", label="Samples/second")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Samples per Second")
    ax1.set_title("IMU Sampling Rate Over Time")
    ax1.grid(True)
    ax1.legend()

    # Plot 2: IMU acceleration values over time
    times = np.linspace(0, results.duration, len(results.imu_readings))
    accel_x = [r.accel_x for r in results.imu_readings]
    accel_y = [r.accel_y for r in results.imu_readings]
    accel_z = [r.accel_z for r in results.imu_readings]
    
    ax2.plot(times, accel_x, label="X")
    ax2.plot(times, accel_y, label="Y")
    ax2.plot(times, accel_z, label="Z")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.set_title("IMU Acceleration Values")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

async def main() -> None:
    """Main entry point for the IMU test."""
    colorlogging.configure()
    logger.warning("Starting IMU Test (test-06)")
    
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            # Run the test
            results = await run_imu_test(kos, duration_seconds=5)
            
            # Visualize results
            plot_results(results)
            
    except Exception:
        logger.exception("Test failed. Ensure Z-Bot is connected via USB and the IP is accessible.")
        raise

if __name__ == "__main__":
    asyncio.run(main())
