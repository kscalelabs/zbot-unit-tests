import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

cur_dir = Path(__file__).parent

# Read the JSON files
with open(cur_dir / 'sim_actuator_states.json', 'r') as f:
    sim_data = json.load(f)
try:
    with open(cur_dir / 'real_actuator_states.json', 'r') as f:
        real_data = json.load(f)
    has_real_data = True
except FileNotFoundError:
    real_data = None
    has_real_data = False

# Convert to pandas DataFrames
sim_df = pd.DataFrame(sim_data)
if has_real_data:
    real_df = pd.DataFrame(real_data)

# Get unique actuator IDs
actuator_ids = sim_df['actuator_id'].unique()

# Create a dictionary to store kp and kd for each actuator
control_params = {}
for actuator_id in actuator_ids:
    actuator_data = sim_df[sim_df['actuator_id'] == actuator_id].iloc[0]
    control_params[actuator_id] = {
        'kp': actuator_data['kp'],
        'kd': actuator_data['kd']
    }

# Create figure with subplots for each actuator
num_actuators = len(actuator_ids)
fig, axes = plt.subplots(num_actuators, 2, figsize=(20, 4*num_actuators))
fig.suptitle(f'Commanded vs Actual Position Over Time for Actuator {actuator_ids[0]}')

# Ensure axes is 2D even with single actuator
if num_actuators == 1:
    axes = axes.reshape(1, 2)

# Plot for each actuator
for i, actuator_id in enumerate(actuator_ids):
    sim_actuator = sim_df[sim_df['actuator_id'] == actuator_id]
    
    # Plot simulation data
    axes[i, 0].plot(sim_actuator['time'], sim_actuator['commanded_position'], 
            label='Commanded', linestyle='--')
    axes[i, 0].plot(sim_actuator['time'], sim_actuator['position'], 
            label='Actual')
    axes[i, 0].set_xlabel('Time (s)')
    axes[i, 0].set_ylabel('Position')
    axes[i, 0].set_title(f'Actuator {actuator_id} - Simulation')
    axes[i, 0].legend()
    axes[i, 0].grid(True)
    
    # Plot real data if available, otherwise leave empty
    if has_real_data:
        real_actuator = real_df[real_df['actuator_id'] == actuator_id]
        axes[i, 1].plot(real_actuator['time'], real_actuator['commanded_position'], 
                        label='Commanded', linestyle='--')
        axes[i, 1].plot(real_actuator['time'], real_actuator['position'], 
                        label='Actual')
        axes[i, 1].set_title(f'Actuator {actuator_id} - Real (kp={control_params[actuator_id]["kp"]}, kd={control_params[actuator_id]["kd"]})')
    else:
        axes[i, 1].set_title('No Real Data Available')
    
    axes[i, 1].set_xlabel('Time (s)')
    axes[i, 1].set_ylabel('Position')
    axes[i, 1].legend()
    axes[i, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
