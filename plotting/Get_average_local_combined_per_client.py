import json
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename


# Function to compute overall mean and standard deviation
def compute_overall_stats(metric_data):
    mean_values = [item['mean'] for item in metric_data]
    std_values = [item['std'] for item in metric_data]

    overall_mean = sum(mean_values) / len(mean_values)
    overall_std = math.sqrt(sum(std ** 2 for std in std_values) / len(std_values))

    return overall_mean, overall_std


# Browse for input JSON file
Tk().withdraw()  # Hide the root tkinter window
input_file = askopenfilename(title="Select JSON File", filetypes=[("JSON Files", "*.json")])

if input_file:
    # Load data from selected JSON file
    with open(input_file, "r") as file:
        data = json.load(file)

    # Compute stats for FScore, Recall, and FPR
    fscore_mean, fscore_std = compute_overall_stats(data["FScore"])
    recall_mean, recall_std = compute_overall_stats(data["Recall"])
    fpr_mean, fpr_std = compute_overall_stats(data["FPR"])

    # Prepare results for saving
    results = {
        "FScore": {"mean": fscore_mean, "std": fscore_std},
        "Recall": {"mean": recall_mean, "std": recall_std},
        "FPR": {"mean": fpr_mean, "std": fpr_std}
    }

    # Browse for output JSON file location
    output_file = asksaveasfilename(
        title="Save Results As", defaultextension=".json",
        filetypes=[("JSON Files", "*.json")]
    )

    if output_file:
        # Save results to the selected output file
        with open(output_file, "w") as file:
            json.dump(results, file, indent=4)

        print(f"Results saved to {output_file}")
    else:
        print("Save operation canceled.")
else:
    print("File selection canceled.")
