from time import time

import numpy as np

# Simulation Parameters
COEF = 0.74  # Stimulus Coefficient
DELTA_T = 2.2  # Reaction Time (s)
V_INIT = 44  # Initial Speed (ft/s)
DECEL_RATES = [-2.0, -3.0, -4.6, -5.0]  # List of deceleration rates (ft/s²)
HEADWAY = -114  # Initial Headway (ft)
N_VEHICLES = 10  # Number of Vehicles
T_TOTAL = 30  # Total Simulation Time (s)
T_STEP = 0.1  # Time Step (s)
N_STEPS = int(T_TOTAL / T_STEP)  # Number of Simulation Steps


def generate_time_vector():
    return np.arange(0, T_TOTAL, T_STEP)


def initialize_matrices():
    """ Initialize acceleration, velocity, and position matrices. """
    a = np.zeros((N_VEHICLES, N_STEPS))
    v = np.zeros((N_VEHICLES, N_STEPS))
    d = np.zeros((N_VEHICLES, N_STEPS))
    return a, v, d


def simulate_lead_vehicle(v, a):
    """ Simulate the motion of the first (lead) vehicle with periodic deceleration and acceleration. """
    v[0, 0] = V_INIT  # Initial speed
    a[0, 0] = 0  # Initial acceleration
    decel_index = 0  # Index to track which deceleration rate to use

    for i in range(1, N_STEPS):
        # Check if it's time to decelerate (every 10 seconds)
        if i % 10 == 0 and i != 0:
            # Decelerate for 5 seconds
            decel_rate = DECEL_RATES[decel_index % len(DECEL_RATES)]  # Cycle through the list
            a[0, i] = decel_rate
            decel_index += 1  # Move to the next deceleration rate for the next phase
        # Check if it's time to accelerate back to the previous speed (after 5 seconds of deceleration)
        elif i % 15 == 0 and i != 0:
            # Accelerate back to the previous speed
            a[0, i] = abs(decel_rate)  # Use the absolute value of the last deceleration rate
        # Maintain constant speed otherwise
        else:
            a[0, i] = 0

        # Update velocity
        v_new = v[0, i - 1] + a[0, i] * T_STEP
        if v_new > 0:
            v[0, i] = v_new
        else:
            v[0, i] = 0  # Stop if velocity reaches 0


def simulate_following_vehicles(v, a):
    """ Simulate the motion of the following vehicles based on the GM Model. """
    for n in range(1, N_VEHICLES):
        v[n, 0] = V_INIT
        for i in range(1, N_STEPS):
            if i < int(DELTA_T / T_STEP) * n:
                v[n, i] = V_INIT
            else:
                v_new = v[n, i - 1] + a[n, i - 1] * T_STEP
                if v_new > 0:
                    v[n, i] = v_new
                    a[n, i] = COEF * (v[n - 1, i - 1] - v[n, i - 1])
                else:
                    v[n, i] = 0
                    a[n, i] = 0


def compute_positions(v):
    """ Compute positions of all vehicles over time using trapezoidal integration. """
    d = np.zeros((N_VEHICLES, N_STEPS))
    for n in range(N_VEHICLES):
        d[n, 0] = n * HEADWAY
        for i in range(1, N_STEPS):
            d[n, i] = d[n, i - 1] + (v[n, i - 1] + v[n, i]) / 2 * T_STEP
    return d


def save_results(time, d,target_file_name):
    """ Save simulation results to a text file for plotting. """
    with open(f"data/{target_file_name}", "w") as f:
        for i in range(N_STEPS):
            f.write(f"{time[i]:.1f}\t" + "\t".join(f"{d[n, i]:.2f}" for n in range(N_VEHICLES)) + "\n")


def main():
    """ Main function to run the car-following simulation. """
    time = generate_time_vector()
    a, v, d = initialize_matrices()

    simulate_lead_vehicle(v, a)
    simulate_following_vehicles(v, a)
    d = compute_positions(v)

    save_results(time, d,'CarFollowing_distance.txt')
    save_results(time, a,'CarFollowing_acceleration.txt')
    save_results(time, v,'CarFollowing_velocity.txt')
    print("Simulation complete! Results saved to 'CarFollowing_Modular.txt'.")


if __name__ == "__main__":
    main()
