import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import json

"""
CODE FOR GENERATING f1-SCORE VS ROUNDS
    for multiple plots
"""
def browse_files_and_plot():
    file_paths = filedialog.askopenfilenames(filetypes=[("JSON Files", "*.json")])
    if file_paths:
        plot_graph(file_paths)


def save_plot():
    save_path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files", "*.svg")])
    if save_path:
        plt.savefig(save_path, format="svg")
        tk.messagebox.showinfo("Save Successful", f"Plot saved to {save_path}")

def plot_graph(file_paths):
    plt.figure(figsize=(8, 6))
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        f1_scores = data['FScore']
        rounds = data['round']
        label = file_path.split("/")[-1].replace(".json", "")
        plt.plot(rounds, f1_scores, marker='o', label=label)

    plt.ylim(0, 1)
    # Customizing the plot
    plt.title("F1 Score vs Rounds", fontsize=10)
    plt.xlabel("Rounds",fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    ax = plt.gca()

    # Adjust axis limits to create space for the arrowhead
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max * 1.1])

    # Add an arrowhead to the y-axis directly on the axis line
    ax.annotate('', xy=(0, y_max * 1.05), xytext=(0, y_min),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                annotation_clip=False)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize = 10)
    save_plot()
    plt.show()

# Main window
root = tk.Tk()
root.title("F1 Score Plotter")
root.geometry("400x200")

browse_files_and_plot()
