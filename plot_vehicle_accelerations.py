import os

import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
data = np.loadtxt('data/CarFollowing.txt')

folder_name = 'acceleration'
# Extract time and position data
time = data[:, 0]
positions = data[:, 1:]

# Calculate velocities (first derivative of position)
velocities = np.diff(positions, axis=0) / np.diff(time)[:, np.newaxis]

# Calculate accelerations (second derivative of position)
accelerations = np.diff(velocities, axis=0) / np.diff(time[1:])[:, np.newaxis]

# Plot accelerations for each vehicle separately with the lead vehicle
lead_vehicle_acceleration = accelerations[:, 0]
for i in range(1, accelerations.shape[1]):
    plt.figure(figsize=(10, 6))
    plt.plot(time[2:], lead_vehicle_acceleration, label='Lead Vehicle', color='blue')
    plt.plot(time[2:], accelerations[:, i], label=f'Vehicle {i+1}', color='orange')
    plt.title(f'Acceleration Comparison: Lead Vehicle vs Vehicle {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (ft/s²)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, f'acceleration_vehicle_{i + 1}.png'))
    plt.close()  # C
    plt.show()
