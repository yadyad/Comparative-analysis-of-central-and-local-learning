import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import os


# Function to open file dialog and select files
def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(filetypes=[("JSON Files", "*.json")])
    return file_paths


# Function to save the plot or data
def save_file():
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        defaultextension=".svg",
        filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
    )
    return save_path


# Load and extract FScore data from JSON files
def load_fscore_data(file_paths):
    all_fscores = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            fscores = [item['mean'] for item in data['FScore']]
            all_fscores.append((os.path.basename(file_path).replace('.json', ''), fscores))
    return all_fscores


# Plot FScore
file_paths = select_files()
if file_paths:
    all_fscores = load_fscore_data(file_paths)

    plt.figure(figsize=(10, 6))
    for file_name, fscores in all_fscores:
        rounds = np.arange(1, len(fscores) + 1)
        plt.plot(rounds, fscores, marker='o', label=file_name)

    plt.title('F1 Score Across Rounds', fontname='Arial', fontsize=15)
    plt.xlabel('Round', fontname='Arial', fontsize=15)
    plt.ylabel('F1 Score', fontname='Arial', fontsize=15)
    plt.xticks(fontname='Arial', fontsize=15)
    plt.yticks(fontname='Arial', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = save_file()
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Plot saved to {save_path}")
    else:
        print("Save operation cancelled.")

    plt.show()
else:
    print("No files selected.")
