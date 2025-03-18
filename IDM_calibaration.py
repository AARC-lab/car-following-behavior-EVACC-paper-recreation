import pandas as pd
import numpy as np
from scipy.optimize import minimize


# Calibration
# Define the IDM model
def idm_model(params, df):
    v0, T, s0, a_max, b = params
    # spacing = [initial_spacing]
    # speed = [initial_speed]
    new_speeds = []
    new_spacings = []
    dt = df['Time'].diff().mean()
    for i in range(len(df)):
        v = df.loc[i, 'Speed Follower']
        v_lead = df.loc[i, 'Speed Leader']
        s = df.loc[i, 'Spacing']
        delta_v = v_lead - v

        # Desired gap
        s_star = s0 + v * T + (v * delta_v) / (2 * np.sqrt(a * b))

        # IDM acceleration
        acc = a * (1 - (v / v0) ** delta - (s_star / s) ** 2)

        # Update speed and spacing
        new_v = v + acc * dt
        new_s = s + (v_lead - new_v) * dt

        new_speeds.append(new_v)
        new_spacings.append(new_s)
    return np.array(new_spacings), np.array(new_speeds)

# Define the objective function (spacing RMSE)
def objective(params, subset):
    simulated_spacing, _ = idm_model(params, subset)

    # Debugging: Print simulated and experimental spacing
    # print("Simulated Spacing:", simulated_spacing)
    # print("Experimental Spacing:", experimental_spacing)
    experimental_spacing = np.array(subset['Spacing'])

    rmse = np.sqrt(np.mean((simulated_spacing - experimental_spacing)**2))
    return rmse


def calibaration_start(df):
    # Split the data into six subsets (each 200 seconds long)
    subset_length = int(200 / 0.02)  # 200 seconds, assuming 0.02-second time steps
    num_subsets = 6
    subsets = [df.iloc[i:i + subset_length].reset_index() for i in range(0, len(df), subset_length)]
    # subsets =
    # subsets = df.head(10000)

    print(f"Number of Subset: {len(subsets)}")
    # Calibrate on each subset
    best_rmse = np.inf
    best_params = None

    for i, subset in enumerate(subsets):
        print(f"-------------------Calibrating on subset {i + 1}-------------------")

        for j in range(len(subset)-1):

            # Extract data for the subset
            time_subset = subset.loc[j,'Time']
            v_lead = subset.loc[j,'Speed Leader']
            v = subset.loc[j,'Speed Follower']
            experimental_spacing = subset.loc[j,'Spacing']
            delta_v = v_lead -v
            # Initial conditions
            # initial_spacing = experimental_spacing_subset.iloc[0]
            # initial_speed = follow_speed_subset.iloc[0]

            # Initial guess for parameters
            #         initial_params = [0.5, 0.5, 2.0, 1.5]

            initial_params = [25, 1.5, 2.0, 0.3, 2]
            # v0, T, s0, a_max, b

            # Bounds for parameters
            #         bounds = [(0, None), (0, None), (0, None), (0, None)]
            bounds = [(25.0, 45.0), (1, 2.5), (1.0, 5.0), (0.5, 2.0), (1.0, 2.0)]
            #         bounds = [(20.0, 30.0), (1.0, 3.0), (1.0, 5.0), (0.5, 2.0), (1.0, 2.0)]

            # Run optimization
            result = minimize(objective, initial_params, subset,
                              bounds=bounds, method='L-BFGS-B')

            # Simulate with the calibrated parameters
            simulated_spacing, _ = idm_model(result.x, subset)

            print(f"Simulated Spacing: {len(simulated_spacing)}")
            # print(f"Experimental Spacing: {len(experimental_spacing_subset)}")
            experimental_spacing = np.array(subset['Spacing'])
            rmse = np.sqrt(np.mean((simulated_spacing - experimental_spacing) ** 2))

            # Check if this is the best model so far
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = result.x

            print(f"Subset {i + 1} RMSE: {rmse:.4f}")
            print(f"Params: {result.x}")
    # print("-------------------------------------------------")
    print(f"Best params: {best_params} \nBest RMSE : {best_rmse}")
    return best_params, best_rmse


def test_and_viz_full_dataset(df, best_params, limit=None):
    # Extract relevant columns
    time = np.arange(0, len(df) * 0.02, 0.02)  # Assuming time increments by 0.02 seconds
    #     time = df['Time'].values
    lead_speed = df['Speed Leader'].values
    follow_speed = df['Speed Follower'].values
    experimental_spacing = df['Spacing'].values  # Assuming this is the spacing

    # Test the best model on the entire dataset
    simulated_spacing_full, simulated_speed_full = idm_model(best_params, time, lead_speed, experimental_spacing[0],
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

    print(time)
    print(experimental_spacing)
    print(simulated_spacing_full)
    # Visualize the results
    import matplotlib.pyplot as plt

    # Plot simulated vs experimental spacing
    plt.figure(figsize=(12, 6))
    plt.plot(time, experimental_spacing, label='Experimental Spacing', color='blue', linestyle='-', linewidth=1)
    plt.plot(time, simulated_spacing_full, label='Simulated Spacing', color='red', linestyle='-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Spacing (m)')
    plt.title(f'Simulated vs Experimental Spacing')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot simulated vs experimental speed (including leader speed)
    plt.figure(figsize=(12, 6))
    plt.plot(time, follow_speed, label='Experimental Follower Speed', color='blue', linestyle='-', linewidth=1)
    plt.plot(time, simulated_speed_full, label='Simulated Follower Speed', color='red', linestyle='-', linewidth=1)
    plt.plot(time, lead_speed, label='Leader Speed', color='green', linestyle='-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Simulated vs Experimental Speed')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    # IDM parameters
    a = 0.3  # Maximum acceleration (m/s^2)
    b = 2  # Comfortable deceleration (m/s^2)
    v0 = 25 * 1609.34 / 3600  # Desired speed (m/s) converted from mph
    s0 = 2.0  # Minimum gap (m)
    T = 1.5  # Desired time headway (s)
    delta = 4  # Acceleration exponent

    data_path = "data/combined_data.csv"

    df = pd.read_csv(data_path)
    # df = df.head(10000)
    # Convert speeds from mph to m/s
    df['Speed Follower'] *= 1609.34 / 3600
    df['Speed Leader'] *= 1609.34 / 3600

    # # Time step (assuming constant time step based on data)
    # dt = df['Time'].diff().mean()
    #
    # # Apply IDM update
    # new_speeds = []
    # new_spacings = []
    #
    # for i in range(len(df) - 1):
    #     v = df.loc[i, 'Speed Follower']
    #     v_lead = df.loc[i, 'Speed Leader']
    #     s = df.loc[i, 'Spacing']
    #     delta_v = v_lead - v
    #
    #     # Desired gap
    #     s_star = s0 + v * T + (v * delta_v) / (2 * np.sqrt(a * b))
    #
    #     # IDM acceleration
    #     acc = a * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    #
    #     # Update speed and spacing
    #     new_v = v + acc * dt
    #     new_s = s + (v_lead - new_v) * dt
    #
    #     new_speeds.append(new_v)
    #     new_spacings.append(new_s)
    #
    # # Add last known values (since there's no update for the last row)
    # new_speeds.append(new_speeds[-1])
    # new_spacings.append(new_spacings[-1])
    #
    # # Add new values to dataframe
    # df['New Speed Follower'] = new_speeds
    # df['New Spacing'] = new_spacings
    #
    # df[['Time', 'Speed Follower', 'Speed Leader','Spacing', 'New Speed Follower', 'New Spacing']]
    # print(0)

    # Split the dataset based on the gap setting

    medium_gap_df = df[df['gap_setting'] == 'Medium']
    short_gap_df = df[df['gap_setting'] == 'Short']
    long_gap_df = df[df['gap_setting'] == 'Long']
    xlong_gap_df = df[df['gap_setting'] == 'XLong']

    medium_gap_best_params,medium_gap_best_rmse = calibaration_start(medium_gap_df)

    test_and_viz_full_dataset(medium_gap_df, medium_gap_best_params, limit=(0, 170))
