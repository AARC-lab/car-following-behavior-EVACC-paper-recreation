import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def kmh_to_ms(speed_kmh):
    return speed_kmh * (1000 / 3600)

def load_best_params(folder_path,gap_setting):
    best_params_list = []
    model_names = ['OVRV','IDM','OVM','CACC']
    # model_names = ['OVRV','IDM']
    for model_name in model_names:
        file_path = f"{folder_path}/{model_name}/best_params.json"

        with open(file_path) as f:
            data = json.load(f)

        best_key = None
        lowest_rmse = float('inf')
        best_params = None

        for key, value in data.items():
            if key == gap_setting:
                best_params = value['best_params']
            # rmse = value['best_rmse']
            # if rmse < lowest_rmse:
            #     lowest_rmse = rmse
            #     best_params = value['best_params']
            #     best_key = key


        best_params_list.append({f"{model_name}": best_params})
    return best_params_list

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

def desired_spacing(v_f, s0, T):
    return s0 + T * v_f

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




def simulate_full_data(df,best_params,model_name):
    simulated_spacing_full, simulated_speed_full = None,None
    # Extract relevant columns
    time = np.arange(0, len(df) * 0.02, 0.02)  # Assuming time increments by 0.02 seconds
    lead_speed = df['Speed Leader'].values
    follow_speed = df['Speed Follower'].values
    experimental_spacing = df['Spacing'].values  # Assuming this is the spacing

    if model_name=="OVRV":
        simulated_spacing_full, simulated_speed_full = ovrv_model(best_params, time, lead_speed, experimental_spacing[0],
                                                              follow_speed[0])

    elif model_name=="IDM":
        simulated_spacing_full, simulated_speed_full = modified_idm_model(best_params, time, lead_speed,
                                                                          experimental_spacing[0],
                                                                          follow_speed[0])
    elif model_name=="OVM":
        simulated_spacing_full, simulated_speed_full = ovm_model(best_params, time, lead_speed, experimental_spacing[0],
                                                                 follow_speed[0])
    else:
        a_leader = np.gradient(lead_speed, time)
        simulated_spacing_full, simulated_speed_full = cacc_model(best_params, time, lead_speed, a_leader, experimental_spacing[0], follow_speed[0])

    return simulated_spacing_full,simulated_speed_full
    # Apply the limit to the data
    # if limit is not None:
    #     start, end = limit
    #     mask = (time >= start) & (time <= end)
    #     time = time[start:end]
    #     experimental_spacing = experimental_spacing[start:end]
    #     simulated_spacing_full = simulated_spacing_full[start:end]
    #     follow_speed = follow_speed[start:end]
    #     simulated_speed_full = simulated_speed_full[start:end]
    #     lead_speed = lead_speed[start:end]







if __name__ == '__main__':
    data_path = "data/combined_data.csv"
    report_dir = 'REPORTS'
    start_index = 0
    end_index = 5000
    # gap_settings = 'medium'
    # gap_settings = 'long'
    gap_settings = 'xlong'
    df = pd.read_csv(data_path)

    df['Speed Follower'] = kmh_to_ms(df['Speed Follower'])
    df['Speed Leader'] = kmh_to_ms(df['Speed Leader'])

    time = np.arange(0, len(df) * 0.02, 0.02)
    time = time[start_index:end_index]

    experimental_spacing = df['Spacing'].values[start_index:end_index]
    lead_speed = df['Speed Leader'].values[start_index:end_index]

    best_params = load_best_params(report_dir,gap_settings)
    # plt.figure(figsize=(12, 6))
    # plt.plot(time,experimental_spacing,label="Experimental",color='blue',linestyle='dotted',linewidth=1)
    # for best_param in best_params:
    #     print(type(best_param))
    #     if "OVRV" in list(best_param.keys())[0]:
    #         ovrv_simulated_spacing,ovrv_simulated_speed = simulate_full_data(df,best_param['OVRV'],"OVRV")
    #         ovrv_simulated_spacing = ovrv_simulated_spacing[start_index:end_index]
    #         ovrv_simulated_speed = ovrv_simulated_speed[start_index:end_index]
    #         plt.plot(time,ovrv_simulated_spacing,label="OVRV",color="green",linestyle='-',linewidth=1)
    #     elif "IDM" in list(best_param.keys())[0]:
    #         IDM_simulated_spacing,IDM_simulated_speed = simulate_full_data(df, best_param['IDM'], "IDM")
    #         IDM_simulated_spacing = IDM_simulated_spacing[start_index:end_index]
    #         IDM_simulated_speed = IDM_simulated_speed[start_index:end_index]
    #         plt.plot(time, IDM_simulated_spacing, label="IDM", color="red", linestyle='-', linewidth=1)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Spacing (m)')
    # plt.title(f'Simulated vs Experimental Spacing')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # print(0)

    # Prepare for plotting
    spacing_curves = []
    speed_curves = []

    # Simulate and collect spacing/speed for each model
    for best_param in best_params:
        model_key = list(best_param.keys())[0]
        model_params = best_param[model_key]

        simulated_spacing, simulated_speed = simulate_full_data(df, model_params, model_key)
        spacing_curves.append((model_key, simulated_spacing[start_index:end_index]))
        speed_curves.append((model_key, simulated_speed[start_index:end_index]))

    # --- Plot Spacing ---
    # Apply clean seaborn theme
    sns.set(style="whitegrid")
    colors = sns.color_palette("Set2", len(spacing_curves))

    # --- Plot Spacing ---
    plt.figure(figsize=(10, 5))
    plt.plot(time, experimental_spacing, label="Experimental", color='black',
             linestyle='--', linewidth=1.5)

    for (model_name, spacing), color in zip(spacing_curves, colors):
        plt.plot(time, spacing, label=model_name, linewidth=1.5, color=color)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Spacing (m)', fontsize=12)
    plt.title(f'Simulated vs Experimental Spacing\n({gap_settings.capitalize()} Gap Setting)', fontsize=14,
              weight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{report_dir}/final_results/{gap_settings}_Spacing.pdf", dpi=300)
    plt.show()

    # --- Plot Speed ---
    plt.figure(figsize=(10, 5))
    plt.plot(time, lead_speed, label="Leader Speed", color='black',
             linestyle='--', linewidth=1.5)

    for (model_name, speed), color in zip(speed_curves, colors):
        plt.plot(time, speed, label=model_name, linewidth=1.5, color=color)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Speed (m/s)', fontsize=12)
    plt.title(f'Simulated vs Experimental Speed\n({gap_settings.capitalize()} Gap Setting)', fontsize=14,
              weight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{report_dir}/final_results/{gap_settings}_Speed.pdf", dpi=300)
    plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(time, experimental_spacing, label="Experimental", color='blue', linestyle='dotted', linewidth=1)
    # for model_name, spacing in spacing_curves:
    #     plt.plot(time, spacing, label=model_name, linewidth=1)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Spacing (m)')
    # plt.title(f'Simulated vs Experimental Spacing - {gap_settings} Gap Setting')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{report_dir}/final_results/{gap_settings}l_Spacing.pdf")
    # plt.show()
    #
    # # --- Plot Speed ---
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, lead_speed, label="Leader Speed", color='blue', linestyle='dotted', linewidth=1)
    # for model_name, speed in speed_curves:
    #     plt.plot(time, speed, label=model_name, linewidth=1)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Speed (m/s)')
    # plt.title(f'Simulated vs Experimental Speed - {gap_settings} Gap Setting')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{report_dir}/final_results/{gap_settings}_Speed.pdf")
    # plt.show()

    print(0)






