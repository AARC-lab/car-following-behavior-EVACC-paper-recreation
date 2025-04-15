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


# Define the objective function (spacing RMSE)
def objective(params, time, lead_speed_subset, experimental_spacing_subset, initial_spacing, initial_speed):
    simulated_spacing, _ = ovrv_model(params, time, lead_speed_subset, initial_spacing, initial_speed)

    #     print(f"Simulated Spacing: {len(simulated_spacing)}")
    #     print(f"Experimental Spacing: {len(experimental_spacing_subset)}")
    rmse = np.sqrt(np.mean((simulated_spacing - experimental_spacing_subset) ** 2))
    return rmse

def ovrv_model(params, time, lead_speed, initial_spacing, initial_speed):
    k1, k2, eta, tau = params
    spacing = [initial_spacing]
    speed = [initial_speed]
    for t in range(1, len(time)):
        dt = time[t] - time[t-1]
        a = k1 * (spacing[-1] - eta - tau * speed[-1]) + k2 * (lead_speed[t-1] - speed[-1])
        v_new = speed[-1] + a * dt
        s_new = spacing[-1] + (lead_speed[t-1] - speed[-1]) * dt
        spacing.append(s_new)
        speed.append(v_new)
    return np.array(spacing), np.array(speed)


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
        lead_speed_subset = subset['Speed Follower'].values
        follow_speed_subset = subset['Speed Follower'].values
        experimental_spacing_subset = subset['Spacing'].values

        # Initial conditions
        initial_spacing = experimental_spacing_subset[0]
        initial_speed = follow_speed_subset[0]

        # Initial guess for parameters
        initial_params = [0.5, 0.5, 2.0, 1.5]

        # Bounds for parameters
        bounds = [(0, None), (0, None), (0, None), (0, None)]

        # Run optimization
        result = minimize(objective, initial_params, args=(
        time_subset, lead_speed_subset, experimental_spacing_subset, initial_spacing, initial_speed),
                          bounds=bounds, method='L-BFGS-B')

        # Simulate with the calibrated parameters
        simulated_spacing, _ = ovrv_model(result.x, time_subset, lead_speed_subset, initial_spacing, initial_speed)

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


def test_and_viz_full_dataset(df, best_params,model_name,report_path,gap_setting, limit=None):
    # Extract relevant columns
    time = np.arange(0, len(df) * 0.02, 0.02)  # Assuming time increments by 0.02 seconds
    lead_speed = df['Speed Leader'].values
    follow_speed = df['Speed Follower'].values
    experimental_spacing = df['Spacing'].values  # Assuming this is the spacing

    # Test the best model on the entire dataset
    simulated_spacing_full, simulated_speed_full = ovrv_model(best_params, time, lead_speed, experimental_spacing[0],
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

    # Plot simulated vs experimental spacing
    plt.figure(figsize=(12, 6))
    plt.plot(time, experimental_spacing, label='Experimental Spacing', color='blue', linestyle='-', linewidth=2)
    plt.plot(time, simulated_spacing_full, label='Simulated Spacing', color='red', linestyle='-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Spacing (m)')
    plt.title(f'Simulated vs Experimental Spacing of {model_name} - {gap_setting}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}/{model_name}/{gap_setting}_spacing.png")
    # plt.show()

    # Plot simulated vs experimental speed (including leader speed)
    plt.figure(figsize=(12, 6))
    # plt.plot(time, follow_speed, label='Experimental Follower Speed', color='blue', linestyle='-', linewidth=2)
    plt.plot(time, simulated_speed_full, label='Simulated Follower Speed', color='red', linestyle='-', linewidth=2)
    plt.plot(time, lead_speed, label='Leader Speed', color='green', linestyle='-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Simulated vs Experimental Speed of {model_name} - {gap_setting}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_path}/{model_name}/{gap_setting}_speed.png")
    # plt.show()



data_path = "../data/combined_data.csv"
df = pd.read_csv(data_path)
df['Speed Follower'] = kmh_to_ms(df['Speed Follower'])
df['Speed Leader'] = kmh_to_ms(df['Speed Leader'])

medium_gap_df = df[df['gap_setting']=='Medium']
short_gap_df = df[df['gap_setting']=='Short']
long_gap_df = df[df['gap_setting']=='Long']
xlong_gap_df = df[df['gap_setting']=='XLong']

medium_gap_best_params,medium_gap_best_rmse = calibaration_start(medium_gap_df)
test_and_viz_full_dataset(medium_gap_df, medium_gap_best_params,model_name="OVRV",report_path="../REPORTS/",gap_setting="medium", limit=(0, 10000))

long_gap_best_params,long_gap_best_rmse = calibaration_start(long_gap_df)
test_and_viz_full_dataset(long_gap_df, long_gap_best_params,model_name="OVRV",report_path="../REPORTS/",gap_setting="long", limit=(0, 10000))

xlong_gap_best_params,xlong_gap_best_rmse = calibaration_start(xlong_gap_df)
test_and_viz_full_dataset(xlong_gap_df, xlong_gap_best_params,model_name="OVRV",report_path="../REPORTS/",gap_setting="xlong", limit=(0, 10000))