import os
import zipfile
import shutil
import pandas as pd

# Define directories
directories = [
    "data",
    "notebooks",
    "src",
    "workflows",
    "artifacts",
    "mlruns"
]

# Create directories
for d in directories:
    os.makedirs(d, exist_ok=True)
    print(f"Created directory: {d}")

# Extract Dataset
zip_path = "reviews_data_dump.zip"
extract_path = "data"

if os.path.exists(zip_path):
    print(f"\nExtracting {zip_path} to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
    
    # Check contents
    extracted_files = os.listdir(extract_path)
    print(f"Files in {extract_path}: {extracted_files}")
    
    # Try to find the CSV and loading it to show initial info (Req 1)
    # The user mentioned "contents multiple folders and CSV files"
    # We'll look for the csv
    csv_file = None
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                break
    
    if csv_file:
        print(f"\nFound CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        print("\n--- Dataset Info ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\n--- Missing Values ---")
        print(df.isnull().sum())
        
        if 'rating' in df.columns or 'Rate' in df.columns: # Adjust based on actual col name
            col = 'rating' if 'rating' in df.columns else 'Rate' # common names
            print("\n--- Rating Distribution ---")
            print(df[col].value_counts())
    else:
        print("No CSV file found in the extracted data.")

else:
    print(f"Error: {zip_path} not found.")
