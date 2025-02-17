import json
import matplotlib.pyplot as plt
import pandas as pd


kp = 80.0
kd = 2.0

# Read the JSON files
with open('sim_actuator_states.json', 'r') as f:
    sim_data = json.load(f)
with open('real_actuator_states.json', 'r') as f:
    real_data = json.load(f)

# Convert to pandas DataFrames
sim_df = pd.DataFrame(sim_data)
real_df = pd.DataFrame(real_data)

# Get unique actuator IDs (assuming they match between sim and real)
actuator_ids = sim_df['actuator_id'].unique()

# Create figure with subplots for each actuator, side by side for sim and real
num_actuators = len(actuator_ids)
fig, axes = plt.subplots(num_actuators, 2, figsize=(20, 4*num_actuators))
fig.suptitle(f'Commanded vs Actual Position Over Time by Actuator (kp={kp}, kd={kd})')

# If there's only one actuator, wrap axes in a 2D array
if num_actuators == 1:
    axes = axes.reshape(1, 2)

# Plot for each actuator
for i, actuator_id in enumerate(actuator_ids):
    sim_actuator = sim_df[sim_df['actuator_id'] == actuator_id]
    real_actuator = real_df[real_df['actuator_id'] == actuator_id]
    
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
    
    # Plot real data
    axes[i, 1].plot(real_actuator['time'], real_actuator['commanded_position'], 
                    label='Commanded', linestyle='--')
    axes[i, 1].plot(real_actuator['time'], real_actuator['position'], 
                    label='Actual')
    axes[i, 1].set_xlabel('Time (s)')
    axes[i, 1].set_ylabel('Position')
    axes[i, 1].set_title(f'Actuator {actuator_id} - Real')
    axes[i, 1].legend()
    axes[i, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
