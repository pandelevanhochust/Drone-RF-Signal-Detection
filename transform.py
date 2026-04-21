import os
import numpy as np
import pandas as pd
from pathlib import Path

input_base = Path("C:/Users/navis/toanlv/DroneDetect_V2/BOTH")
output_base = Path("C:/Users/navis/toanlv/OutputHdf5/")

folders = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON", "INS_FY", "INS_HO", "INS_ON",
    "MIN_FY", "MIN_HO", "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY", "MP2_HO",
    "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON"
]


def transform_data():
    for folder_name in folders:
        source_dir = input_base / folder_name
        target_dir = output_base / folder_name

        # 1. Create target directory safely
        target_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            print(f"Skipping {folder_name}: Directory not found.")
            continue

        # 2. Iterate through files
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if not filename.endswith('.dat'):
                    continue

                file_path = Path(root) / filename
                print(f"Processing: {file_path}")

                try:
                    # 3. Read Binary Data
                    # Using 'with' ensures the file is closed even if an error occurs
                    with open(file_path, "rb") as f_in:
                        data = np.fromfile(f_in, dtype="float32", count=240000000)

                    # 4. Transform IQ Data
                    # Convert to complex then back to float32 view
                    data = data.astype(np.float32).view(np.complex64).view(np.float32)

                    # 5. Normalize (Memory efficient)
                    # np.std is faster than sqrt(var)
                    data = (data - np.mean(data)) / np.std(data)

                    # 6. Split into 400 chunks (5ms segments)
                    chunks = np.array_split(data, 400)
                    del data  # Free memory early

                    # 7. Create DataFrame
                    df = pd.DataFrame(chunks)

                    # 8. Save to HDF5
                    # Using explicit keywords to avoid the Positional Argument TypeError
                    output_file = target_dir / f"{filename[:11]}.h5"
                    df.to_hdf(str(output_file), key='data', mode='w')

                    print(f"   -> Saved: {output_file.name}")

                    # Final cleanup for the loop iteration
                    del df
                    del chunks

                except Exception as e:
                    print(f"   [!] Error processing {filename}: {e}")
        print(f"Completed folder: {folder_name}\n")


if __name__ == "__main__":
    transform_data()