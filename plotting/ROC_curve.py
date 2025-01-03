import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tkinter import Tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename

# Set global font settings to Arial with font size 10
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 15

# Open a file dialog to select files
Tk().withdraw()  # Hide the root Tkinter window
filepaths = askopenfilenames(title="Select JSON Files", filetypes=[("JSON Files", "*.json")])

# Initialize a plot
plt.figure(figsize=(12, 8))

# Loop through the selected JSON files
for filepath in filepaths:
    # Load the JSON data
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Extract FPR, TPR, and AUC
    fpr = data.get("fpr", [])
    tpr = data.get("tpr", [])
    auc = data.get("auc", None)

    # Extract file name without extension
    filename = filepath.split("/")[-1].replace('.json', '')  # Remove .json extension

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f"{filename} (AUC = {auc:.3f})")

# Plot customization
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess", fontsize=15)
plt.xlabel("False Positive Rate (FPR)", fontsize=15)
plt.ylabel("True Positive Rate (TPR)", fontsize=15)
plt.title("ROC Curves for different algorithms", fontsize=15)
plt.legend(loc="lower right", fontsize=15)
plt.grid(alpha=0.5)
plt.tight_layout()

# Save the plot as SVG
save_filepath = asksaveasfilename(
    title="Save Plot As",
    defaultextension=".svg",
    filetypes=[("SVG Files", "*.svg")]
)

if save_filepath:  # Save only if a file was selected
    plt.savefig(save_filepath, format='svg')
    print(f"Plot saved to {save_filepath}")

# Show the plot
plt.show()
