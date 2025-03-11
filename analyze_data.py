import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    """Load the simulation data from the text file."""
    data = np.loadtxt(file_path)
    time = data[:, 0]  # First column is time
    positions = data[:, 1:]  # Remaining columns are positions of vehicles
    return time, positions


def compute_velocities(time, positions):
    """Compute velocities from positions using finite differences."""
    velocities = np.zeros_like(positions)
    for n in range(positions.shape[1]):  # Iterate over each vehicle
        velocities[:, n] = np.gradient(positions[:, n], time)
    return velocities


def compute_accelerations(time, velocities):
    """Compute accelerations from velocities using finite differences."""
    accelerations = np.zeros_like(velocities)
    for n in range(velocities.shape[1]):  # Iterate over each vehicle
        accelerations[:, n] = np.gradient(velocities[:, n], time)
    return accelerations


def plot_acceleration_comparison(time, lead_acceleration, vehicle_acceleration, vehicle_number, output_dir):
    """Plot and save acceleration comparison between lead vehicle and another vehicle."""
    plt.figure(figsize=(10, 6))
    plt.plot(time, lead_acceleration, label="Lead Vehicle", color="blue")
    plt.plot(time, vehicle_acceleration, label=f"Vehicle {vehicle_number}", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (ft/s²)")
    plt.title(f"Acceleration Comparison: Lead Vehicle vs Vehicle {vehicle_number}")
    plt.legend()
    plt.grid()

    # Save the figure to the output directory
    output_path = os.path.join(output_dir, f"acceleration_vehicle_{vehicle_number}.png")
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory


def main(file_path):
    """Main function to load data, compute accelerations, and plot."""
    # Create output directory for saving figures
    output_dir = "data/acceleration"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    time, positions = load_data(file_path)

    # Compute velocities
    velocities = compute_velocities(time, positions)

    # Compute accelerations
    accelerations = compute_accelerations(time, velocities)

    # Plot and save figures for each vehicle compared with the lead vehicle
    lead_acceleration = accelerations[:, 0]  # Lead vehicle acceleration
    for n in range(1, accelerations.shape[1]):  # Skip the lead vehicle itself (index 0)
        plot_acceleration_comparison(
            time,
            lead_acceleration,
            accelerations[:, n],
            vehicle_number=n + 1,
            output_dir=output_dir,
        )

    print(f"Figures saved in the '{output_dir}' folder.")


# Example usage
if __name__ == "__main__":
    file_path = "data/CarFollowing_acc_dec.txt"  # Path to the generated data file
    main(file_path)
