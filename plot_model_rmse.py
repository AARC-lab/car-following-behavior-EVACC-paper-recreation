import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


plt.rcParams['font.family'] = 'Serif'

def load_rmse_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_gap_rmse(data, gap_type):
    return {k.replace(f'_{gap_type}', ''): v for k, v in data.items() if k.endswith(f'_{gap_type}')}

def plot_rmse_bar(rmse_dict,ml_rmse, title, save_path):

    # models = list(rmse_dict.keys())
    # values = list(rmse_dict.values())
    rmse_dict['RF'] = ml_rmse
    sorted_items = sorted(rmse_dict.items(), key=lambda x: x[1])
    models, values = zip(*sorted_items)

    # Set a clean Seaborn style
    sns.set(style="whitegrid")

    # Create figure
    plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Serif"
    bars = plt.bar(models, values, color=sns.color_palette("dark", len(models)))

    # Add text labels above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/4, yval + 0.2, f'{yval:.4f}',
                 ha='center', va='bottom', fontsize=16)

    # Labeling and aesthetics
    plt.xlabel('Model', fontsize=18)
    plt.ylabel('RMSE (m)', fontsize=18)
    plt.title(title, fontsize=18, weight='bold',pad=20)
    plt.xticks(rotation=30, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    sns.despine()

    # Save the figure
    plt.savefig(save_path, dpi=200)
    plt.show()


def main(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    data = load_rmse_json(json_path)
    ml_rmse = {"medium":0.0046,"long":0.0016,"xlong":0.0025}

    for gap in ['medium', 'long', 'xlong']:
        rmse_subset = extract_gap_rmse(data, gap)
        if rmse_subset:
            plot_rmse_bar(rmse_subset,ml_rmse[gap], f"RMSE for {gap.capitalize()} Gap", f"REPORTS/final_results/modified/updated_rmse_{gap}.pdf")

if __name__ == '__main__':
    json_path = 'REPORTS/rmse.json'
    main(json_path)
