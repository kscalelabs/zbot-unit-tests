import argparse
import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pykos import KOS

logger = logging.getLogger(__name__)


class IMUVisualizer:
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.initialize_plot()
        plt.show()  # Show the window

    def initialize_plot(self):
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])

        self.update_arrows(np.eye(3))
        plt.title("IMU Orientation")

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

        return Rz @ Ry @ Rx

    def update_arrows(self, R):
        origin = np.array([0, 0, 0])

        if hasattr(self, "arrows"):
            for arrow in self.arrows:
                arrow.remove()

        self.arrows = [
            self.ax.quiver(origin[0], origin[1], origin[2], R[0, i], R[1, i], R[2, i], color=c, length=0.5)
            for i, c in enumerate(["r", "g", "b"])
        ]

    def update_visualization(self, roll, pitch, yaw):
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        self.update_arrows(R)

        if hasattr(self, "text"):
            self.text.remove()
        self.text = self.ax.text2D(
            0.02, 0.95, f"Roll: {roll:6.1f}°\nPitch: {pitch:6.1f}°\nYaw: {yaw:6.1f}°", transform=self.ax.transAxes
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


async def run_visualization(kos: KOS, update_rate: float = 0.05):
    viz = IMUVisualizer()

    try:
        # Test connection
        logger.info("Testing IMU connection...")
        await kos.imu.get_imu_values()
        logger.info("IMU connection successful!")

        # Zero the IMU
        logger.info("Zeroing IMU...")
        await kos.imu.zero(duration=2.0)
        logger.info("IMU zeroed. Starting visualization...")

        while True:
            try:
                # Get values
                euler_data = await kos.imu.get_euler_angles()

                # Update visualization
                viz.update_visualization(euler_data.roll, euler_data.pitch, euler_data.yaw)

                # Short sleep to prevent overwhelming the system
                await asyncio.sleep(update_rate)

            except Exception as e:
                logger.error(f"Error reading IMU data: {e}")
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nStopping visualization...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        plt.close("all")  # Close all plot windows


async def main():
    parser = argparse.ArgumentParser(description="IMU 3D Visualization")
    parser.add_argument("--host", type=str, default="100.89.14.31", help="Robot IP address")
    parser.add_argument("--port", type=int, default=50051, help="Robot port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Connecting to robot at {args.host}:{args.port}")

    async with KOS(ip=args.host, port=args.port) as kos:
        await run_visualization(kos)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        plt.close("all")
