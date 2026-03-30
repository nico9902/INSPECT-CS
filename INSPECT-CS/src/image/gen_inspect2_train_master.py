import pandas as pd
import os
import random

# Path to the folder containing split files
split_root_folder = "data/folds/unimodal_reports/"

# Function to generate mortality data and save to separate files
def generate_mortality_files():
    # Iterate over the mortality types
    for mortality_type in ["1_month_mortality", "6_month_mortality", "12_month_mortality"]:
        data = {
            "impression_id": [],
            mortality_type: [],
            "split": []
        }

        # Read split information from the files in the split folder
        split_folder = os.path.join(split_root_folder, mortality_type)
        for split_file in os.listdir(split_folder):
            if split_file.endswith(".csv"):  # Ensure only CSV files are processed
                split_path = os.path.join(split_folder, split_file)
                split_df = pd.read_csv(split_path)

                # Extract the split type from the file name (e.g., "train", "valid", "test")
                split_type = os.path.splitext(split_file)[0]  # Remove the file extension

                # For each row in the split file, generate random mortality values and assign the split
                for j in range(len(split_df)):
                    data["impression_id"].append(split_df["impression_id"].iloc[j])  
                    data[mortality_type].append(split_df[mortality_type].iloc[j])  
                    data["split"].append(split_type)

        # Create a DataFrame for the current mortality type
        mortality_df = pd.DataFrame(data)

        # Save the mortality file to CSV
        output_file = f"{mortality_type}.csv"
        mortality_df.to_csv(output_file, index=False)
        print(f"CSV file '{output_file}' has been generated successfully.")

# Generate the files
generate_mortality_files()