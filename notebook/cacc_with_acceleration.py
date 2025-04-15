import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import json


def kmh_to_ms(speed_kmh):
    return speed_kmh * (1000 / 3600)

# --- CACC model without leader acceleration ---
def desired_spacing(v_f, s0, T):
    return s0 + T * v_f

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


def cacc_model(params, time, v_leader, a_leader, initial_spacing, initial_speed):
    k1, k2, k3, s0, T = params
    spacing = [initial_spacing]
    speed = [initial_speed]

    for t in range(1, len(time)):
        dt = time[t] - time[t - 1]
        s = spacing[-1]
        v_f = speed[-1]
        v_l = v_leader[t - 1]
        a_l = a_leader[t - 1]

        s_des = desired_spacing(v_f, s0, T)
        a = k1 * (v_l - v_f) + k2 * (s - s_des) + k3 * a_l

        v_new = max(v_f + a * dt, 0)
        s_new = max(s + (v_l - v_f) * dt, 0)

        speed.append(v_new)
        spacing.append(s_new)

    return np.array(spacing), np.array(speed)

def objective_cacc(params, time, v_leader, a_leader, spacing_actual, initial_spacing, initial_speed):
    spacing_sim, _ = cacc_model(params, time, v_leader, a_leader, initial_spacing, initial_speed)
    rmse = np.sqrt(np.mean((spacing_sim - spacing_actual) ** 2))
    return rmse

def calibrate_cacc(df, subset_duration=200, dt=0.02):
    subset_length = int(subset_duration / dt)
    num_subsets = 6
    subsets = [df.iloc[i * subset_length:(i + 1) * subset_length] for i in range(num_subsets)]

    best_rmse = np.inf
    best_params = None

    for i, subset in enumerate(subsets):
        print(f"\n--- Calibrating on subset {i + 1}/{num_subsets} ---")
        time = subset['Time'].values
        v_leader = subset['Speed Leader'].values
        spacing = subset['Spacing'].values
        v_follower = subset['Speed Follower'].values

        a_leader = np.gradient(v_leader, time)  # estimate leader acceleration

        initial_spacing = spacing[0]
        initial_speed = v_follower[0]

        # Initial guess: [k1, k2, k3, s0, T]
        initial_params = [0.5, 0.2, 0.1, 2.0, 1.5]
        bounds = [(0.1, 2.0), (0.1, 2.0), (0.0, 2.0), (0.5, 10.0), (0.5, 3.0)]

        result = minimize(objective_cacc, initial_params,
                          args=(time, v_leader, a_leader, spacing, initial_spacing, initial_speed),
                          bounds=bounds, method='L-BFGS-B')

        rmse = objective_cacc(result.x, time, v_leader, a_leader, spacing, initial_spacing, initial_speed)
        print(f"Subset {i + 1} RMSE: {rmse:.4f}")
        print(f"Params: {result.x}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = result.x

    print("\n--- Best Calibration Results ---")
    print(f"Best Params: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")

    return best_params


# --- Evaluation and Visualization ---
def test_and_viz_full_dataset(df, best_params, model_name, report_path, limit=None):
    time = np.arange(0, len(df) * 0.02, 0.02)
    v_leader = df['Speed Leader'].values
    v_follower = df['Speed Follower'].values
    spacing_actual = df['Spacing'].values

    # spacing_sim, speed_sim = cacc_model(best_params, time, v_leader, spacing_actual[0], v_follower[0])
    a_leader = np.gradient(v_leader, time)
    spacing_sim, speed_sim = cacc_model(best_params, time, v_leader, a_leader, spacing_actual[0], v_follower[0])

    # spacing_sim, speed_sim = cacc_model(best_params, time, v_leader, spacing_actual[0], v_follower[0])
    rmse = np.sqrt(np.mean((spacing_sim - spacing_actual) ** 2))
    print("-----------------------------------------------------------------------------")
    print(f"Full dataset RMSE: {rmse:.4f}")
    update_dict_from_file("../REPORTS/rmse.json", model_name, rmse)

    with open(f'../REPORTS/best_params_{model_name}.json', 'w') as f:
        json.dump({
            'best_params': best_params.tolist(),
            'best_rmse': float(rmse)
        }, f, indent=4)

    if limit is not None:
        start, end = limit
        time = time[start:end]
        spacing_actual = spacing_actual[start:end]
        spacing_sim = spacing_sim[start:end]
        v_follower = v_follower[start:end]
        speed_sim = speed_sim[start:end]
        v_leader = v_leader[start:end]

    plt.figure(figsize=(12, 6))
    plt.plot(time, spacing_actual, label='Actual Spacing', color='blue')
    plt.plot(time, spacing_sim, label='Predicted Spacing', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Spacing (m)')
    plt.title(f'CACC - Simulated vs Actual Spacing')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}{model_name}_spacing.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time, speed_sim, label='Predicted Follower Speed', color='red', linestyle='--')
    plt.plot(time, v_leader, label='Leader Speed', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title(f'CACC - Simulated Follower Speed vs Leader Speed')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}{model_name}_speed.png")
    plt.show()


df = pd.read_csv("../data/combined_data.csv")
df['Speed Follower'] = kmh_to_ms(df['Speed Follower'])
df['Speed Leader'] = kmh_to_ms(df['Speed Leader'])

medium_gap_df = df[df['gap_setting'] == 'Medium']
best_params = calibrate_cacc(medium_gap_df)
test_and_viz_full_dataset(medium_gap_df, best_params, model_name="CACC", report_path="../REPORTS/CACC/", limit=(0, 10000))
