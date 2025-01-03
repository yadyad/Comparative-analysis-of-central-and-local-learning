import os
import re
import numpy as np
from numpy import array  # Ensure 'array' is defined for eval
from tkinter import Tk, filedialog
import ast
import json
from math import inf  # Import 'inf' for eval context



"""
    CODE TO GET F1-SCORE;FPR;RECALL FROM LOGS
"""
# Function to browse and select files
def browse_files():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Log Files in Different Folders")
    return list(file_paths)


# Function to extract metrics from a file
def extract_metrics(file_path):
    #metrics_pattern = re.compile(r"metrics (\{.*?\})")    #local
    #metrics_pattern = re.compile(r"metrics is (\{.*?\})") #central
    metrics_pattern = re.compile(r" Global Model Test Results:(\{.*?\})")  #federated
    metrics_list = []

    with open(file_path, 'r') as file:
        for line in file:
            match = metrics_pattern.search(line)
            if match:
                metrics_str = match.group(1)
                try:
                    metrics_dict = eval(metrics_str,
                                        {"array": array, "np": np, "inf": inf})  # Provide safe context for eval
                    metrics_list.append(metrics_dict)
                except Exception as e:
                    print(f"Error parsing metrics: {e}")

    return metrics_list


def save_aggregated_results(data):
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.asksaveasfilename(
        title="Save Aggregated Metrics as JSON",
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if file_path:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Aggregated metrics saved to {file_path}")


# Function to compute mean and standard deviation of metrics among clients
def compute_stats_among_clients(metrics_data_across_files):
    if not metrics_data_across_files:
        return {}

    aggregated_metrics = {
        'FScore': [],
        'Recall': [],
        'FPR': [],
        'TPR': []
    }

    # Align metrics for each client (line index) across files
    for client_idx in range(len(metrics_data_across_files[0])):
        client_fscore_values = []
        client_recall_values = []
        client_fpr_values = []
        client_tpr_values = []

        for file_metrics in metrics_data_across_files:
            client_metrics = file_metrics[client_idx]
            client_fscore_values.append(client_metrics.get('FScore', np.nan))
            client_recall_values.append(client_metrics.get('recall', np.nan))

            fpr = client_metrics.get('fpr', None)
            if fpr is not None and len(fpr) > 2:
                client_fpr_values.append(fpr[1])

            tpr = client_metrics.get('tpr', None)
            if tpr is not None and len(tpr) > 2:
                client_tpr_values.append(tpr[1])

        aggregated_metrics['FScore'].append(
            {'mean': np.nanmean(client_fscore_values), 'std': np.nanstd(client_fscore_values)})
        aggregated_metrics['Recall'].append(
            {'mean': np.nanmean(client_recall_values), 'std': np.nanstd(client_recall_values)})
        aggregated_metrics['FPR'].append({'mean': np.nanmean(client_fpr_values), 'std': np.nanstd(client_fpr_values)})
        aggregated_metrics['TPR'].append({'mean': np.nanmean(client_tpr_values), 'std': np.nanstd(client_tpr_values)})

    return aggregated_metrics


# Main script
if __name__ == "__main__":
    print("Please select log files from different folders:")
    file_paths = browse_files()

    metrics_data_across_files = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        file_metrics = extract_metrics(file_path)
        metrics_data_across_files.append(file_metrics)

    if metrics_data_across_files:
        client_stats = compute_stats_among_clients(metrics_data_across_files)
        print("\nMetrics Statistics Among Clients:")
        for idx, (fscore, recall, fpr, tpr) in enumerate(
                zip(client_stats['FScore'], client_stats['Recall'], client_stats['FPR'], client_stats['TPR'])):
            print(f"Client {idx + 1}:")
            print(f"  FScore - Mean: {fscore['mean']}, Std: {fscore['std']}")
            print(f"  Recall - Mean: {recall['mean']}, Std: {recall['std']}")
            print(f"  FPR - Mean: {fpr['mean']}, Std: {fpr['std']}")
            print(f"  TPR - Mean: {tpr['mean']}, Std: {tpr['std']}")
        save_aggregated_results(client_stats)
    else:
        print("No metrics data found in the selected files.")
