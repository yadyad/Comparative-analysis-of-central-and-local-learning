import json
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

"""
CODE TO GENERATE MULTIPLE BOX PLOT FOR FPR
"""

# Open file selection window
def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_paths = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json")])
    return file_paths

# Open save location window
def select_save_location():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    save_path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files", "*.svg")])
    return save_path

# Load data from JSON files
def load_json_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data.append((os.path.basename(file_path).replace('.json', ''), json.load(f)))
    return data


def split_label(label, max_words=2):
    label = label.replace(' ', '\n')
    return label

# Plot the data
def plot_data(data, output_file):
    num_plots = len(data)
    rows = (num_plots + 1) // 2
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = axes.flatten()

    for i, (filename, dataset) in enumerate(data):
        ax = axes[i]
        x = dataset["x"]
        y = dataset["all_means"]
        stds = dataset["all_stds"]
        labels = dataset["group_labels"]
        labels = [split_label(label) for label in labels]
        ax.bar(x, y, yerr=stds, capsize=5, alpha=0.7, color="#97C139", edgecolor="black")
        ax.set_title(filename)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha="center")
        ax.set_ylabel("FPR")
        ax.set_xlabel("Techniques")
        ax.set_ylim(0, 0.4)

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig(output_file, format="svg")
    print(f"Plot saved as {output_file}")
    plt.show()

# Main execution
if __name__ == "__main__":
    file_paths = select_files()
    if not file_paths:
        print("No files selected.")
    else:
        data = load_json_files(file_paths)
        save_location = select_save_location()
        if save_location:
            plot_data(data, output_file=save_location)
        else:
            print("Save location not selected.")
