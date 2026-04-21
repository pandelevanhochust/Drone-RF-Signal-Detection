import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

# --- STEP 1: Tell the code where your data is ---
# Change these two lines to match your computer folders
source_base_path = Path("C:/Users/navis/toanlv/OutputHdf5")
save_results_path = Path("C:/Users/navis/toanlv/PCA_Results")
save_results_path.mkdir(exist_ok=True)

folders = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON", "INS_FY", "INS_HO", "INS_ON",
    "MIN_FY", "MIN_HO", "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY", "MP2_HO",
    "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON"
]

for folder_name in folders:
    folder_address = source_base_path / folder_name

    if not folder_address.exists():
        continue

    print(f"Working on: {folder_name}")

    # --- STEP 2: Collect all the file addresses ("paths") ---
    all_file_paths = list(folder_address.glob("*.h5"))

    # --- STEP 3: Load and stack the data ---
    data_list = []
    for file_path in all_file_paths:
        # 'file_path' is the "address" the computer uses to find the file
        df_single = pd.read_hdf(file_path, key='data')
        data_list.append(df_single)

    if data_list:
        # Stack all files into one big matrix
        big_df = pd.concat(data_list, axis=0)

        # --- STEP 4: Run the PCA ---
        # PCA needs to know how many 'components' (features) to keep.
        # We use min() to make sure we don't ask for more than the data has.
        n_comp = min(big_df.shape[0], big_df.shape[1], 6000)

        pca = PCA(n_components=n_comp)
        reduced_data = pca.fit_transform(big_df)

        # --- STEP 5: Save the result ---
        output_df = pd.DataFrame(reduced_data)
        output_df.to_hdf(save_results_path / f"{folder_name}_PCA.h5", key='pca_data', mode='w')
        print(f"Successfully reduced {folder_name} to {n_comp} features.")