import asyncio
import pykos
from dataclasses import dataclass

@dataclass
class Actuator:
    actuator_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str
    calibration_offset: float  # degrees to back off from limit
    direction: int = 1  # 1 for positive direction, -1 for negative direction

ACTUATOR_LIST: list[Actuator] = [
    # Actuator(31, 85.0, 3.0, 5.0, "left_hip_pitch_04", 10.0, 1),
    # Actuator(32, 85.0, 2.0, 5.0, "left_hip_roll_03", 10.0, 1),
    # Actuator(33, 30.0, 2.0, 5.0, "left_hip_yaw_03", 10.0, 1),
    # Actuator(34, 60.0, 2.0, 5.0, "left_knee_04", 10.0, 1),
    # Actuator(35, 80.0, 1.0, 5.0, "left_ankle_02", 10.0, 1),

    # Actuator(41, 85.0, 3.0, 5.0, "right_hip_pitch_04", 10.0, 1),
    # Actuator(42, 85.0, 2.0, 5.0, "right_hip_roll_03", 10.0, 1),
    # Actuator(43, 30.0, 2.0, 5.0, "right_hip_yaw_03", 10.0, 1),
    Actuator(44, 1000.0, 2.0, 10.0, "right_knee_04", 120.0, 1),
    Actuator(45, 80.0, 1.0, 10.0, "right_ankle_02", 37.0, 1),
]

async def calibrate_actuator(kos: pykos.KOS, actuator: Actuator) -> None:
    print(f"\nCalibrating {actuator.joint_name} (ID: {actuator.actuator_id}) in {'positive' if actuator.direction > 0 else 'negative'} direction")
    
    # Configure actuator with lower torque and gains for safe calibration
    await kos.actuator.configure_actuator(
        actuator_id=actuator.actuator_id,
        kp=actuator.kp,
        kd=actuator.kd,
        max_torque=actuator.max_torque,
        torque_enabled=True
    )
    
    # Move slowly until blocked
    position_history = []
    last_position = (await kos.actuator.get_actuators_state([actuator.actuator_id])).states[0].position
    while True:
        await kos.actuator.command_actuators([{
            'actuator_id': actuator.actuator_id,
            'position': last_position + (0.5 * actuator.direction)
        }])
        await asyncio.sleep(0.02)  # Wait for movement
        
        state = await kos.actuator.get_actuators_state([actuator.actuator_id])
        current_position = state.states[0].position
        print(f"Current position: {current_position}")
        
        # Keep track of last 10 positions
        position_history.append(current_position)
        if len(position_history) > 10:
            position_history.pop(0)
        
        # If position hasn't changed significantly over last 10 readings, we've hit the limit
        if len(position_history) == 10 and max(position_history) - min(position_history) < 0.005:
            break
            
        last_position = current_position
    
    print(f"Found limit at position: {current_position}, offset: {actuator.calibration_offset}, difference: {current_position - actuator.calibration_offset * actuator.direction}")
    
    # Move back to offset position slowly
    target_position = current_position - (actuator.calibration_offset * actuator.direction)
    current_command = current_position
    while abs(current_command - target_position) > 0.5:
        # Move in 0.5 degree increments
        if (actuator.direction > 0 and current_command > target_position) or \
           (actuator.direction < 0 and current_command < target_position):
            current_command -= (0.5 * actuator.direction)
        
        await kos.actuator.command_actuators([{
            'actuator_id': actuator.actuator_id,
            'position': current_command
        }])
        await asyncio.sleep(0.02)  # Short delay between commands
    
    await kos.actuator.configure_actuator(
        actuator_id=actuator.actuator_id,
        kp=actuator.kp * 2,
        kd=actuator.kd,
        torque_enabled=True
    )
    # Final position command
    await kos.actuator.command_actuators([{
        'actuator_id': actuator.actuator_id,
        'position': target_position
    }])
    await asyncio.sleep(1)  # Wait for final movement to complete
    
    # Set new calibration offset
    await kos.actuator.configure_actuator(
        actuator_id=actuator.actuator_id,
        zero_position=True
    )
    
    # Disable torque
    await kos.actuator.configure_actuator(
        actuator_id=actuator.actuator_id,
        torque_enabled=False
    )

async def main():
    kos = pykos.KOS()
    kos.connect()
    
    try:
        # Calibrate each actuator in sequence
        for actuator in ACTUATOR_LIST:
            await calibrate_actuator(kos, actuator)
        
        print("\nCalibration complete for all actuators")
    
    except KeyboardInterrupt:
        print("\nCalibration interrupted! Disabling torque on all actuators...")
        # Disable torque on all actuators
        for actuator in ACTUATOR_LIST:
            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                torque_enabled=False
            )
        print("All actuators disabled")

asyncio.run(main())