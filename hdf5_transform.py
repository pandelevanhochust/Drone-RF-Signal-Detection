import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Point this to the folder shown in your image
input_folder = Path("data")
# Point this to where you want the .h5 files to go
output_folder = Path("output")

# Ensure the output directory exists
output_folder.mkdir(parents=True, exist_ok=True)


def process_data_folder():
    # Loop through every .dat file in that specific folder
    for file_path in input_folder.glob("*.dat"):
        print(f"Processing: {file_path.name}...")

        try:
            # 1. Read Binary Data
            # count=240,000,000 float32s is ~915MB of RAM
            data = np.fromfile(file_path, dtype="float32", count=240000000)

            # 2. Convert to Complex then back to float32 view
            # (Matches your original logic for IQ data handling)
            data = data.view(np.complex64).view(np.float32)

            # 3. Normalization (In-place to save RAM)
            mean_val = np.mean(data)
            std_val = np.std(data)
            data = (data - mean_val) / std_val

            # 4. Split into 400 chunks (5ms segments)
            chunks = np.array_split(data, 400)

            # 5. Convert to DataFrame and Export
            df = pd.DataFrame(chunks)

            # Save using the first 11 characters of the filename
            output_filename = f"{file_path.stem[:11]}.h5"
            save_path = output_folder / output_filename

            df.to_hdf(save_path, key='data', mode='w')
            print(f"Done! Saved to: {save_path}")

            # Cleanup for the next file in the loop
            del data
            del chunks
            del df

        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")


if __name__ == "__main__":
    process_data_folder()