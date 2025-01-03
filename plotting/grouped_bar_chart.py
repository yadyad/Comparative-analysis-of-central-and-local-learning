import json
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt


# Function to load JSON data from a file
def load_json_file():
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, 'r') as f:
            return json.load(f), file_path.split("/")[-1]  # Return data and the filename
    return None, None


def split_label(label, max_words=2):
    label = label.replace(' ', '\n')
    return label

def save_plot_as_svg(fig):
    save_path = filedialog.asksaveasfilename(
        defaultextension=".svg",
        filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
    )
    if save_path:
        fig.savefig(save_path, format='svg')
        print(f"Plot saved as SVG at {save_path}")
    else:
        print("Save operation cancelled.")


# Function to plot grouped bar chart
def plot_grouped_bar_chart(data1, data2, label1, label2):
    x = np.arange(len(data1["group_labels"]))  # the label locations
    width = 0.35  # width of bars
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    # Define different shades of green
    label1 = label1.replace('.json', '')
    label2 = label2.replace('.json', '')
    color1 = '#97C139'  # Lighter green
    color2 = 'green'  # Darker green
    Type1 = label1.split('_')[0]
    setting1 = label1.split('_')[1]
    setting2 = label2.split('_')[1]
    group_labels = [split_label(label) for label in data1["group_labels"]]
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, data1["all_means"], width, yerr=data1["all_stds"],
                   label=setting1, color=color1, capsize=5)
    bars2 = ax.bar(x + width / 2, data2["all_means"], width, yerr=data2["all_stds"],
                   label=setting2, color=color2, capsize=5)

    # Labels and title
    ax.set_xlabel('Algorithms', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title(f'{Type1} (Mean ± Std) IID vs Non-IID', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels,fontsize=12)
    ax.legend(fontsize=12)

    # Adding grid for better visualization
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_plot_as_svg(fig)
    plt.show()


# Main application
def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    print("Select the first JSON file.")
    data1, label1 = load_json_file()
    if not data1:
        print("No file selected for the first JSON.")
        return

    print("Select the second JSON file.")
    data2, label2 = load_json_file()
    if not data2:
        print("No file selected for the second JSON.")
        return

    plot_grouped_bar_chart(data1, data2, label1, label2)


if __name__ == "__main__":
    main()
