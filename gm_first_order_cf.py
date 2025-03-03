import numpy as np

# Simulation Parameters
COEF = 0.74  # Stimulus Coefficient
DELTA_T = 2.2  # Reaction Time (s)
V_INIT = 44  # Initial Speed (ft/s)
DECEL_RATE = -4.6  # Deceleration Rate of the Lead Vehicle (ft/s²)
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
    """ Simulate the motion of the first (lead) vehicle. """
    v[0, 0] = V_INIT
    a[0, 0] = DECEL_RATE

    for i in range(1, N_STEPS):
        v_new = v[0, i - 1] + a[0, i - 1] * T_STEP
        if v_new > 0:
            v[0, i] = v_new
            a[0, i] = DECEL_RATE
        else:
            v[0, i] = 0
            a[0, i] = 0


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


def save_results(time, d):
    """ Save simulation results to a text file for plotting. """
    with open("data/CarFollowing_Modular.txt", "w") as f:
        for i in range(N_STEPS):
            f.write(f"{time[i]:.1f}\t" + "\t".join(f"{d[n, i]:.2f}" for n in range(N_VEHICLES)) + "\n")


def main():
    """ Main function to run the car-following simulation. """
    time = generate_time_vector()
    a, v, d = initialize_matrices()

    simulate_lead_vehicle(v, a)
    simulate_following_vehicles(v, a)
    d = compute_positions(v)

    save_results(time, d)
    print("Simulation complete! Results saved to 'CarFollowing_Modular.txt'.")


if __name__ == "__main__":
    main()
