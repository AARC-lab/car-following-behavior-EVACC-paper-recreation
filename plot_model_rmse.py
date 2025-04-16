import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def load_rmse_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_gap_rmse(data, gap_type):
    return {k.replace(f'_{gap_type}', ''): v for k, v in data.items() if k.endswith(f'_{gap_type}')}

# def plot_rmse_bar(rmse_dict, title, save_path):
#     models = list(rmse_dict.keys())
#     values = list(rmse_dict.values())
#
#     plt.figure(figsize=(6, 4))
#     bars = plt.bar(models, values)
#
#     # Add RMSE values on top of each bar
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}',
#                  ha='center', va='bottom', fontsize=10)
#
#     plt.xlabel('Model')
#     plt.ylabel('RMSE')
#     plt.title(title)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.grid(True, axis='y')
#     plt.savefig(save_path)
#     plt.show()

def plot_rmse_bar(rmse_dict, title, save_path):

    # models = list(rmse_dict.keys())
    # values = list(rmse_dict.values())
    sorted_items = sorted(rmse_dict.items(), key=lambda x: x[1])
    models, values = zip(*sorted_items)

    # Set a clean Seaborn style
    sns.set(style="whitegrid")

    # Create figure
    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, values, color=sns.color_palette("Set2", len(models)))

    # Add text labels above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}',
                 ha='center', va='bottom', fontsize=10)

    # Labeling and aesthetics
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('RMSE (m)', fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.xticks(rotation=30, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    sns.despine()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.show()


def main(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    data = load_rmse_json(json_path)

    for gap in ['medium', 'long', 'xlong']:
        rmse_subset = extract_gap_rmse(data, gap)
        if rmse_subset:
            plot_rmse_bar(rmse_subset, f"RMSE for {gap.capitalize()} Gap", f"REPORTS/rmse_{gap}.pdf")

if __name__ == '__main__':
    json_path = 'REPORTS/rmse.json'
    main(json_path)
