import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tkinter import Tk, filedialog
from tkinter.filedialog import askopenfilenames, askdirectory

# Function to select multiple files
def select_files():
    Tk().withdraw()  # Prevent the Tkinter root window from appearing
    file_paths = askopenfilenames(
        title="Select JSON Files",
        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
    )
    return file_paths

# Function to select directory to save plots
def select_directory():
    Tk().withdraw()  # Prevent the Tkinter root window from appearing
    directory_path = askdirectory(title="Select Directory to Save Plots")
    return directory_path

def save_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(title="Save JSON File", defaultextension=".json", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
    return file_path

# Main function
def plot_metrics_from_files():
    file_paths = select_files()
    if not file_paths:
        print("No files selected.")
        return

    save_directory = select_directory()
    if not save_directory:
        print("No directory selected.")
        return

    metrics = ['FScore', 'Recall', 'FPR']  # Metrics to combine in one plot
    grouped_data = defaultdict(list)  # To group files by algorithm type

    # Group files based on their starting name (algorithm type)
    for file_path in file_paths:
        file_name = file_path.split('/')[-1]  # Extract file name
        group_name = file_name.replace('.json', '')  # Remove .json from the file name
        grouped_data[group_name].append(file_path)

    # Define the desired order of algorithms
    desired_order = ['Local Learning', 'Central Learning', 'Federated Averaging', 'FedProx', 'SCAFFOLD', 'FedNova']
    #desired_order = ['Default', 'Random oversampling', 'Weighted oversampling', 'Augmentation']
    # Process and plot each metric for all groups
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Prepare data for the current metric
        group_labels = []  # To label each group on the x-axis
        all_means = []  # To store means of the current metric for all groups
        all_stds = []  # To store stds of the current metric for all groups

        # Process groups in the desired order
        for group in desired_order:
            if group in grouped_data:
                files = grouped_data[group]
                group_labels.append(group)
                group_means = []
                group_stds = []

                # Extract the third value of means and stds for the current metric from all files in the group
                for file_path in files:
                    print(file_path)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    try:
                        group_means.append(data[metric][2]['mean'])  # Use the third value
                        group_stds.append(data[metric][2]['std'])   # Use the third value
                    except IndexError:
                        group_means.append(data[metric][0]['mean'])
                        group_stds.append(data[metric][0]['std'])
                # Use the third value directly for the current metric
                all_means.append(group_means[0])  # Assuming all files in a group are consistent
                all_stds.append(group_stds[0])

        # Plot the combined bar plot for the current metric
        x = np.arange(len(group_labels))  # the label locations
        width = 0.4  # width of the bars

        plt.bar(x, all_means, width, yerr=all_stds, capsize=5, alpha=0.8, color='#97C139')
        results = {"x": x.tolist(), "all_means": all_means, "all_stds": all_stds, "group_labels": group_labels}
        output_file_path = save_file()
        if not output_file_path:
            print("No save location selected. Exiting.")
            return
        with open(output_file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        if metric == 'FScore':
            plt.title(f'F1Score (Mean ± Std) Across Algorithms', fontname='Arial', fontsize=15)
        else:
            plt.title(f'{metric} (Mean ± Std) Across Algorithms', fontname='Arial', fontsize=15)
        plt.ylabel('Values', fontname='Arial', fontsize=15)
        plt.xlabel('Algorithms', fontname='Arial', fontsize=15)
        plt.xticks(x, group_labels, ha='center', fontname='Arial', fontsize=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot as an SVG file
        plot_filename = f"{save_directory}/{metric}_plot.svg"
        plt.tight_layout()
        plt.savefig(plot_filename, format='svg')
        plt.show()

# Run the program
if __name__ == "__main__":
    plot_metrics_from_files()
