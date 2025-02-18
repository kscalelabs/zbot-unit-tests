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
        plt.title("IMU Orientation with Balance Control")

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

    def update_visualization(self, roll, pitch, yaw, balance_adjustments=None):
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        self.update_arrows(R)

        if hasattr(self, "text"):
            self.text.remove()

        text = f"Roll: {roll:6.1f}°\nPitch: {pitch:6.1f}°\nYaw: {yaw:6.1f}°"

        if balance_adjustments:
            text += f"\n\nBalance Adjustments:\n"
            text += f'Left Hip: {balance_adjustments["left_hip"]:6.3f}\n'
            text += f'Right Hip: {balance_adjustments["right_hip"]:6.3f}\n'
            text += f'Left Ankle: {balance_adjustments["left_ankle"]:6.3f}\n'
            text += f'Right Ankle: {balance_adjustments["right_ankle"]:6.3f}\n'

        self.text = self.ax.text2D(0.02, 0.95, text, transform=self.ax.transAxes)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class BalanceController:
    def __init__(self):
        # PID parameters - adjust these for best performance
        self.kp_roll = 0.05  # Proportional gain for roll
        self.kd_roll = 0.02  # Derivative gain for roll
        self.ki_roll = 0.001  # Integral gain for roll

        self.kp_pitch = 0.05  # Proportional gain for pitch
        self.kd_pitch = 0.02  # Derivative gain for pitch
        self.ki_pitch = 0.001  # Integral gain for pitch

        # Target angles (usually 0 for standing upright)
        self.target_roll = 0.0
        self.target_pitch = 0.0

        # Previous error values for derivative calculation
        self.prev_roll_error = 0.0
        self.prev_pitch_error = 0.0

        # Accumulated error for integral calculation
        self.roll_error_sum = 0.0
        self.pitch_error_sum = 0.0

        # Last timestamp for dt calculation
        self.last_time = None

    def calculate_adjustments(self, roll, pitch, current_time):
        # Calculate time delta
        if self.last_time is None:
            dt = 0.02  # Default value (50Hz)
        else:
            dt = current_time - self.last_time
        self.last_time = current_time

        # Calculate errors
        roll_error = self.target_roll - roll
        pitch_error = self.target_pitch - pitch

        # Calculate error derivatives
        if dt > 0:
            roll_error_derivative = (roll_error - self.prev_roll_error) / dt
            pitch_error_derivative = (pitch_error - self.prev_pitch_error) / dt
        else:
            roll_error_derivative = 0
            pitch_error_derivative = 0

        # Update previous errors
        self.prev_roll_error = roll_error
        self.prev_pitch_error = pitch_error

        # Update error sums for integral (with anti-windup)
        self.roll_error_sum = np.clip(self.roll_error_sum + roll_error * dt, -10, 10)
        self.pitch_error_sum = np.clip(self.pitch_error_sum + pitch_error * dt, -10, 10)

        # Calculate PID outputs
        roll_adjustment = (
            self.kp_roll * roll_error + self.kd_roll * roll_error_derivative + self.ki_roll * self.roll_error_sum
        )

        pitch_adjustment = (
            self.kp_pitch * pitch_error + self.kd_pitch * pitch_error_derivative + self.ki_pitch * self.pitch_error_sum
        )

        # Calculate motor adjustments based on PID outputs
        left_hip_adjustment = -pitch_adjustment - roll_adjustment
        right_hip_adjustment = -pitch_adjustment + roll_adjustment
        left_ankle_adjustment = pitch_adjustment - roll_adjustment
        right_ankle_adjustment = pitch_adjustment + roll_adjustment

        return {
            "left_hip": left_hip_adjustment,
            "right_hip": right_hip_adjustment,
            "left_ankle": left_ankle_adjustment,
            "right_ankle": right_ankle_adjustment,
        }


async def apply_balance_adjustments(kos, adjustments):
    """Apply balance adjustments to robot actuators.

    This function maps the calculated adjustments to specific actuator commands
    for the robot. You may need to adjust actuator_ids based on your robot configuration.
    """
    try:
        # Scale the adjustments to appropriate values for your robot
        scale_factor = 0.1  # This value needs tuning for your specific robot

        # Map adjustments to actuator commands
        commands = [
            {"actuator_id": 31, "position_offset": adjustments["left_hip"] * scale_factor},  # left_hip_pitch
            {"actuator_id": 41, "position_offset": adjustments["right_hip"] * scale_factor},  # right_hip_pitch
            {"actuator_id": 35, "position_offset": adjustments["left_ankle"] * scale_factor},  # left_ankle
            {"actuator_id": 45, "position_offset": adjustments["right_ankle"] * scale_factor},  # right_ankle
        ]

        # Send commands to robot
        await kos.actuator.command_actuators(commands)

    except Exception as e:
        logger.error(f"Error applying balance adjustments: {e}")


async def run_balance_controller(kos: KOS, update_rate: float = 0.02, apply_control: bool = False):
    viz = IMUVisualizer()
    controller = BalanceController()

    try:
        # Test connection to IMU
        logger.info("Testing IMU connection...")
        try:
            euler_data = await kos.imu.get_euler_angles()
            logger.info(
                f"IMU connection successful! Initial orientation: Roll={euler_data.roll:.1f}°, "
                f"Pitch={euler_data.pitch:.1f}°, Yaw={euler_data.yaw:.1f}°"
            )
        except Exception as e:
            logger.error(f"Failed to connect to IMU: {e}")
            return

        # Zero the IMU
        logger.info("Zeroing IMU...")
        await kos.imu.zero(duration=2.0)
        logger.info("IMU zeroed. Starting balance controller...")

        while True:
            try:
                # Get current time for PID calculation
                current_time = asyncio.get_event_loop().time()

                # Get IMU data
                euler_data = await kos.imu.get_euler_angles()

                # Calculate balance adjustments
                adjustments = controller.calculate_adjustments(euler_data.roll, euler_data.pitch, current_time)

                # Apply adjustments to robot if enabled
                if apply_control:
                    await apply_balance_adjustments(kos, adjustments)

                # Update visualization
                viz.update_visualization(euler_data.roll, euler_data.pitch, euler_data.yaw, adjustments)

                # Sleep until next control cycle
                await asyncio.sleep(update_rate)

            except Exception as e:
                logger.error(f"Error in balance control loop: {e}")
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nStopping balance controller...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        plt.close("all")


async def main():
    parser = argparse.ArgumentParser(description="IMU-based Balance Controller")
    parser.add_argument("--host", type=str, default="localhost", help="Simulation host address")
    parser.add_argument("--port", type=int, default=50051, help="Simulation port number")
    parser.add_argument(
        "--update-rate", type=float, default=0.02, help="Control update rate in seconds (default: 0.02 = 50Hz)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--apply-control", action="store_true", help="Actually apply control signals to the robot")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Connecting to simulation at {args.host}:{args.port}")
    if args.apply_control:
        logger.info("Balance controller will apply control signals to the robot")
    else:
        logger.info("Balance controller in monitoring mode (no control signals will be sent)")

    async with KOS(ip=args.host, port=args.port) as kos:
        await run_balance_controller(kos, args.update_rate, args.apply_control)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        plt.close("all")
