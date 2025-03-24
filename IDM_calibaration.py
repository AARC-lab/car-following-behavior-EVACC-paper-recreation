import math

import pandas as pd
import numpy as np
from scipy.optimize import minimize

import numpy as np
import math


def modified_idm_model(params, time, lead_speed, initial_spacing, initial_speed):
    """
    Stable IDM model with checks for inf/nan and negative delta_v.
    """
    v0, T, s0, a_max, b = params
    spacing = [initial_spacing]
    speed = [initial_speed]
    eps = 1e-6  # Small constant to avoid division by zero

    for t in range(1, len(time)):
        dt = time[t] - time[t - 1]
        delta_v = lead_speed[t - 1] - speed[-1]

        # --- Compute s_star with constraints ---
        s_star = s0 + speed[-1] * T + (speed[-1] * delta_v) / (2 * np.sqrt(a_max * b))
        s_star = max(s_star, s0)  # Never allow s_star < s0 (minimum jam distance)

        # --- Handle negative delta_v (follower faster than leader) ---
        if delta_v < 0:
            # Reduce the impact of negative delta_v to prevent extreme braking
            s_star = s0 + speed[-1] * T  # Ignore the (negative) dynamic term

        # --- Stabilize acceleration calculation ---
        spacing_denominator = max(spacing[-1], eps)  # Avoid division by zero
        a = a_max * (1 - (speed[-1] / v0) ** 4 - (s_star / spacing_denominator) ** 2)

        # --- Limit acceleration to physical bounds ---
        a = np.clip(a, -b, a_max)  # Deceleration cannot exceed comfortable braking (b)

        # --- Update speed and spacing ---
        v_new = max(speed[-1] + a * dt, 0)  # Speed cannot be negative
        s_new = max(spacing[-1] + (lead_speed[t - 1] - speed[-1]) * dt, 0)  # Spacing cannot be negative

        spacing.append(s_new)
        speed.append(v_new)

    return np.array(spacing), np.array(speed)


# Define the IDM model
def idm_model(params, time, lead_speed, initial_spacing, initial_speed):
    v0, T, s0, a_max, b = params
    spacing = [initial_spacing]
    speed = [initial_speed]
    for t in range(1, len(time)):
        dt = time[t] - time[t - 1]
        leader_speed = lead_speed[t-1]
        follower_speed = speed[-1]
        delta_v = lead_speed[t - 1] - speed[-1]
        # Compute desired spacing
        s_star = s0 + speed[-1] * T + (speed[-1] * delta_v) / (2 * np.sqrt(a_max * b))
        if s_star<0:
            print(0)
        elif math.isnan(s_star):
            print(0)

        # print(f"Time: {time[t]:.2f}, Spacing: {spacing[-1]:.2f}, Speed: {speed[-1]:.2f}, s_star: {s_star:.2f}")
        # Compute acceleration with stability check
        # Add a small constant to avoid division by zero
        # spacing_denominator = max(spacing[-1], 1e-6)
        # a = a_max * (1 - (speed[-1] / v0) ** 4 - (s_star / spacing_denominator) ** 2)
        a = a_max * (1 - (speed[-1] / v0) ** 4 - (s_star / spacing[-1]) ** 2)
        # Update speed and spacing
        v_new = speed[-1] + a * dt
        if v_new<0:
            print(f"New velocity : {v_new}")
        # Ensure spacing does not become negative
        spacing[-1] = max(spacing[-1], 0)
        s_new = spacing[-1] + (lead_speed[t - 1] - speed[-1]) * dt

        spacing.append(s_new)
        speed.append(v_new)
    return np.array(spacing), np.array(speed)

# Define the objective function (spacing RMSE)
def objective(params, time, lead_speed, experimental_spacing, initial_spacing, initial_speed):
    # simulated_spacing, _ = idm_model(params, time, lead_speed, initial_spacing, initial_speed)
    simulated_spacing, _ = modified_idm_model(params, time, lead_speed, initial_spacing, initial_speed)
    rmse = np.sqrt(np.mean((simulated_spacing - experimental_spacing)**2))
    return rmse


def calibaration_start(df):
    # Split the data into six subsets (each 200 seconds long)
    subset_length = int(200 / 0.02)  # 200 seconds, assuming 0.02-second time steps
    num_subsets = 6
    subsets = [df.iloc[i * subset_length:(i + 1) * subset_length] for i in range(num_subsets)]

    print(f"Number of Subset: {len(subsets)}")
    # Calibrate on each subset
    best_rmse = np.inf
    best_params = None

    for i, subset in enumerate(subsets):
        print(f"-------------------Calibrating on subset {i + 1}-------------------")

        # Extract data for the subset
        time_subset = subset['Time'].values
        lead_speed_subset = subset['Speed Leader'].values
        follow_speed_subset = subset['Speed Follower'].values
        experimental_spacing_subset = subset['Spacing'].values

        # Initial conditions
        initial_spacing = experimental_spacing_subset[0]
        initial_speed = follow_speed_subset[0]

        # Initial guess for parameters
        #         initial_params = [0.5, 0.5, 2.0, 1.5]

        #         initial_params = [25, 1.0, 1.0, 1.0, 1.67]
        # initial_params = [v_0, time_headway, inital_sapcing, max_acc, comportable_braking_dec]
        initial_params = [14, 1.5, 2.0, 2.03, 2]

        # Bounds for parameters
        #         bounds = [(0, None), (0, None), (0, None), (0, None)]
        # Example bounds for IDM parameters [v0, T, s0, a_max, b]
        bounds = [
            (20.0, 40.0),  # v0: Desired speed (m/s)
            (0.5, 2.5),  # T: Safe time headway (s)
            (1.0, 5.0),  # s0: Minimum jam distance (m)
            (0.5, 2.0),  # a_max: Maximum acceleration (m/s²)
            (1.0, 2.0)  # b: Comfortable deceleration (m/s²)
        ]
        #         bounds = [(20.0, 30.0), (1.0, 3.0), (1.0, 5.0), (0.5, 2.0), (1.0, 2.0)]

        # Run optimization
        result = minimize(objective, initial_params, args=(
        time_subset, lead_speed_subset, experimental_spacing_subset, initial_spacing, initial_speed),
                          bounds=bounds, method='L-BFGS-B')

        # Simulate with the calibrated parameters
        # simulated_spacing, _ = idm_model(result.x, time_subset, lead_speed_subset, initial_spacing, initial_speed)
        simulated_spacing, _ = modified_idm_model(result.x, time_subset, lead_speed_subset, initial_spacing, initial_speed)

        print(f"Simulated Spacing: {len(simulated_spacing)}")
        print(f"Experimental Spacing: {len(experimental_spacing_subset)}")
        rmse = np.sqrt(np.mean((simulated_spacing - experimental_spacing_subset) ** 2))

        # Check if this is the best model so far
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = result.x

        print(f"Subset {i + 1} RMSE: {rmse:.4f}")
        print(f"Params: {result.x}")
    print("-------------------------------------------------")
    print(f"Best params: {best_params} \nBest RMSE : {best_rmse}")
    return best_params, best_rmse


def test_and_viz_full_dataset(df, best_params,model_name,report_path, limit=None):
    # Extract relevant columns
    time = np.arange(0, len(df) * 0.02, 0.02)  # Assuming time increments by 0.02 seconds
    #     time = df['Time'].values
    lead_speed = df['Speed Leader'].values
    follow_speed = df['Speed Follower'].values
    experimental_spacing = df['Spacing'].values  # Assuming this is the spacing

    # Test the best model on the entire dataset
    # simulated_spacing_full, simulated_speed_full = idm_model(best_params, time, lead_speed, experimental_spacing[0],
    #                                                          follow_speed[0])
    simulated_spacing_full, simulated_speed_full = modified_idm_model(best_params, time, lead_speed, experimental_spacing[0],
                                                             follow_speed[0])
    rmse_full = np.sqrt(np.mean((simulated_spacing_full - experimental_spacing) ** 2))
    print("-----------------------------------------------------------------------------")
    print(f"Full dataset RMSE: {rmse_full:.4f}")

    # Apply the limit to the data
    if limit is not None:
        start, end = limit
        mask = (time >= start) & (time <= end)
        time = time[mask]
        experimental_spacing = experimental_spacing[mask]
        simulated_spacing_full = simulated_spacing_full[mask]
        follow_speed = follow_speed[mask]
        simulated_speed_full = simulated_speed_full[mask]
        lead_speed = lead_speed[mask]

    # Visualize the results
    import matplotlib.pyplot as plt

    # Plot simulated vs experimental spacing
    plt.figure(figsize=(12, 6))
    plt.plot(time, experimental_spacing, label='Experimental Spacing', color='blue', linestyle='-', linewidth=1)
    plt.plot(time, simulated_spacing_full, label='Simulated Spacing', color='red', linestyle='-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Spacing (m)')
    plt.title(f'Simulated vs Experimental Spacing of {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}{model_name}_spacing.png")
    plt.show()

    # Plot simulated vs experimental speed (including leader speed)
    plt.figure(figsize=(12, 6))
    plt.plot(time, follow_speed, label='Experimental Follower Speed', color='blue', linestyle='-', linewidth=1)
    plt.plot(time, simulated_speed_full, label='Simulated Follower Speed', color='red', linestyle='-', linewidth=1)
    plt.plot(time, lead_speed, label='Leader Speed', color='green', linestyle='-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Simulated vs Experimental Speed of {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}{model_name}_speed.png")
    plt.show()


def kmh_to_ms(column):
    return column * (5 / 18)


if __name__ == '__main__':
    data_path = "data/combined_data.csv"
    report_path = "REPORTS/IDM/"
    model_name = "IDM"
    df = pd.read_csv(data_path)
    df['Speed Follower'] = kmh_to_ms(df['Speed Follower'])
    df['Speed Leader'] = kmh_to_ms(df['Speed Leader'])


    # Divide the dataset based on the gap setting
    medium_gap_df = df[df['gap_setting'] == 'Medium']
    short_gap_df = df[df['gap_setting'] == 'Short']
    long_gap_df = df[df['gap_setting'] == 'Long']
    xlong_gap_df = df[df['gap_setting'] == 'XLong']
    print(medium_gap_df.shape,short_gap_df.shape,long_gap_df.shape,xlong_gap_df.shape)
    medium_gap_best_params, medium_gap_best_rmse = calibaration_start(medium_gap_df)
    print(0)
    test_and_viz_full_dataset(medium_gap_df, medium_gap_best_params,model_name,report_path, limit=(0, 170))
    print(0)
