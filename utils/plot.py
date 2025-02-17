import json
import matplotlib.pyplot as plt
import pandas as pd

# Read the JSON file
with open('actuator_states.json', 'r') as f:
    data = json.load(f)

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# Get unique actuator IDs from the data
actuator_ids = df['actuator_id'].unique()

# Create figure with subplots for each actuator
num_actuators = len(actuator_ids)
fig, axes = plt.subplots(num_actuators, 1, figsize=(12, 4*num_actuators))
fig.suptitle('Commanded vs Actual Position Over Time by Actuator')

# If there's only one actuator, wrap axes in a list for consistent indexing
if num_actuators == 1:
    axes = [axes]

# Plot for each actuator
for i, actuator_id in enumerate(actuator_ids):
    df_actuator = df[df['actuator_id'] == actuator_id]
    axes[i].plot(df_actuator['time'], df_actuator['commanded_position'], 
                 label='Commanded Position', linestyle='--')
    axes[i].plot(df_actuator['time'], df_actuator['position'], 
                 label='Actual Position')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Position')
    axes[i].set_title(f'Actuator {actuator_id} in KOS-Sim')
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
