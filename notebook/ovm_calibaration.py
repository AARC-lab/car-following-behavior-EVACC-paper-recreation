import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import json


def kmh_to_ms(speed_kmh):
    return speed_kmh * (1000 / 3600)

def update_dict_from_file(path, key, value):

    # Load existing dictionary or initialize empty one
    if os.path.exists(path):
        with open(path, 'r') as f:
            dictionary = json.load(f)
    else:
        dictionary = {}

    # Update or add entry
    if key in dictionary:
        print(f"Key '{key}' exists. Updating value to '{value}'.")
    else:
        print(f"Key '{key}' not found. Adding key with value '{value}'.")

    dictionary[key] = value

    # Save back to file
    with open(path, 'w') as f:
        json.dump(dictionary, f, indent=4)

    print("Dictionary successfully updated and saved.")

# --- OVM model implementation ---
def optimal_velocity(s, s_st, vmax):
    """Optimal velocity function (can be sigmoid or linear)."""
    return vmax * np.tanh(s / s_st - 2)

def ovm_model(params, time, lead_speed, initial_spacing, initial_speed):
    vmax, alpha, s_st = params
    spacing = [initial_spacing]
    speed = [initial_speed]

    for t in range(1, len(time)):
        dt = time[t] - time[t - 1]
        s = spacing[-1]
        v = speed[-1]

        v_opt = optimal_velocity(s, s_st, vmax)
        a = alpha * (v_opt - v)

        # Euler integration
        v_new = max(v + a * dt, 0)
        s_new = max(s + (lead_speed[t - 1] - v) * dt, 0)

        speed.append(v_new)
        spacing.append(s_new)

    return np.array(spacing), np.array(speed)

# --- Objective function (RMSE between simulated and experimental spacing) ---
def objective_ovm(params, time, lead_speed, experimental_spacing, initial_spacing, initial_speed):
    simulated_spacing, _ = ovm_model(params, time, lead_speed, initial_spacing, initial_speed)
    rmse = np.sqrt(np.mean((simulated_spacing - experimental_spacing) ** 2))
    return rmse

# --- Calibration function ---
def calibrate_ovm(df, subset_duration=200, dt=0.02):
    subset_length = int(subset_duration / dt)
    num_subsets = 6
    subsets = [df.iloc[i * subset_length:(i + 1) * subset_length] for i in range(num_subsets)]

    best_rmse = np.inf
    best_params = None

    for i, subset in enumerate(subsets):
        print(f"\n--- Calibrating on subset {i + 1}/{num_subsets} ---")

        time = subset['Time'].values
        lead_speed = subset['Speed Leader'].values
        spacing = subset['Spacing'].values
        follower_speed = subset['Speed Follower'].values

        initial_spacing = spacing[0]
        initial_speed = follower_speed[0]

        initial_params = [20.0, 0.5, 10.0]  # [vmax, alpha, s_st]
        bounds = [(5, 35), (0.1, 2.0), (5.0, 20.0)]

        result = minimize(objective_ovm, initial_params,
                          args=(time, lead_speed, spacing, initial_spacing, initial_speed),
                          bounds=bounds, method='L-BFGS-B')

        rmse = objective_ovm(result.x, time, lead_speed, spacing, initial_spacing, initial_speed)
        print(f"Subset {i + 1} RMSE: {rmse:.4f}")
        print(f"Params: {result.x}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = result.x

    print("\n--- Best Calibration Results ---")
    print(f"Best Params: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")
    return best_params

def test_and_viz_full_dataset(df, best_params,model_name,report_path,gap_setting, limit=None):
    # Extract relevant columns
    time = np.arange(0, len(df) * 0.02, 0.02)  # Assuming time increments by 0.02 seconds
    #     time = df['Time'].values
    lead_speed = df['Speed Leader'].values
    follow_speed = df['Speed Follower'].values
    experimental_spacing = df['Spacing'].values  # Assuming this is the spacing

    # Test the best model on the entire dataset
    # simulated_spacing_full, simulated_speed_full = idm_model(best_params, time, lead_speed, experimental_spacing[0],
    #                                                          follow_speed[0])
    simulated_spacing_full, simulated_speed_full = ovm_model(best_params, time, lead_speed, experimental_spacing[0],
                                                             follow_speed[0])
    rmse = np.sqrt(np.mean((simulated_spacing_full - experimental_spacing) ** 2))
    print("-----------------------------------------------------------------------------")
    print(f"Full dataset RMSE: {rmse:.4f}")
    model_gap = f"{model_name}_{gap_setting}"
    update_dict_from_file("../REPORTS/rmse.json", model_gap, rmse)

    os.makedirs(f"../REPORTS/{model_name}", exist_ok=True)

    best_params_dict = {
        'best_params': best_params.tolist(),
        'best_rmse': float(rmse)
    }

    update_dict_from_file(f'../REPORTS/{model_name}/best_params.json', gap_setting, best_params_dict)

    # Apply the limit to the data
    if limit is not None:
        start, end = limit
        mask = (time >= start) & (time <= end)
        time = time[start:end]
        experimental_spacing = experimental_spacing[start:end]
        simulated_spacing_full = simulated_spacing_full[start:end]
        follow_speed = follow_speed[start:end]
        simulated_speed_full = simulated_speed_full[start:end]
        lead_speed = lead_speed[start:end]

        # Visualize the results

    # Plot simulated vs experimental spacing
    plt.figure(figsize=(12, 6))
    plt.plot(time, experimental_spacing, label='Experimental Spacing', color='blue', linestyle='-', linewidth=1)
    plt.plot(time, simulated_spacing_full, label='Simulated Spacing', color='red', linestyle='-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Spacing (m)')
    plt.title(f'Simulated vs Experimental Spacing - {model_name} {gap_setting}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}/{model_name}/{gap_setting}_spacing.pdf")
    # plt.show()

    # Plot simulated vs experimental speed (including leader speed)
    plt.figure(figsize=(12, 6))
    #     plt.plot(time, follow_speed, label='Experimental Follower Speed', color='blue', linestyle='-', linewidth=1)
    plt.plot(time, simulated_speed_full, label='Simulated Follower Speed', color='red', linestyle='-', linewidth=1)
    plt.plot(time, lead_speed, label='Leader Speed', color='green', linestyle='-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Simulated vs Experimental Speed - {model_name} {gap_setting}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}/{model_name}/{gap_setting}_speed.pdf")
    # plt.show()

df = pd.read_csv("../data/combined_data.csv")
df['Speed Follower'] = kmh_to_ms(df['Speed Follower'])
df['Speed Leader'] = kmh_to_ms(df['Speed Leader'])

medium_gap_df = df[df['gap_setting'] == 'Medium']
best_params = calibrate_ovm(medium_gap_df)
test_and_viz_full_dataset(medium_gap_df, best_params, model_name="OVM", report_path="../REPORTS/",gap_setting="medium", limit=(0, 10000))


long_gap_df = df[df['gap_setting'] == 'Long']
best_params = calibrate_ovm(long_gap_df)
test_and_viz_full_dataset(long_gap_df, best_params, model_name="OVM", report_path="../REPORTS/",gap_setting="long", limit=(0, 10000))

xlong_gap_df = df[df['gap_setting'] == 'XLong']
best_params = calibrate_ovm(xlong_gap_df)
test_and_viz_full_dataset(xlong_gap_df, best_params, model_name="OVM", report_path="../REPORTS/",gap_setting="xlong", limit=(0, 10000))
