import re
import json
import numpy as np
from tkinter import Tk, filedialog

# Function to browse and select the log file
"""
CODE TO GET F1-SCORE PER ROUND FROM LOG FILES
"""
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select Log File")
    return file_path

# Function to browse and select the output location
def save_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(title="Save JSON File", defaultextension=".json", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
    return file_path

# Regular expression to extract the required information
pattern = r"Global Model Test Results:\{.*?'FScore': np\.float64\((.*?)\).*?\}"

# Main function
def extract_f1_scores():
    # Select the log file
    log_file_path = select_file()
    if not log_file_path:
        print("No file selected. Exiting.")
        return

    # Lists to store results
    fscores = []
    rounds = []

    # Read and process the log file
    with open(log_file_path, 'r') as file:
        content = file.read()

    # Find all matches for F1 score and associate them with rounds
    matches = re.finditer(pattern, content)
    for round_num, match in enumerate(matches, start=1):
        f1_score = float(match.group(1))  # Convert the string to a float
        fscores.append(f1_score)
        rounds.append(round_num)

    # Prepare the JSON structure
    results = {"fscore": fscores, "round": rounds}

    # Select the output file location
    output_file_path = save_file()
    if not output_file_path:
        print("No save location selected. Exiting.")
        return

    # Save results as a JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Extracted F1 scores and saved to: {output_file_path}")

if __name__ == "__main__":
    extract_f1_scores()
