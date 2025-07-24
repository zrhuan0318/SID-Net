# examples/demo.py

import numpy as np
import pandas as pd
from sidnet import sid_decompose, sid_to_network_df, build_sid_network

# Generate synthetic data: 5 species, 1000 samples
np.random.seed(42)
species_names = [f"Species_{i}" for i in range(5)]
data = np.random.rand(5, 1000)

# Choose a target variable
target_index = 0
Y = np.vstack([data[target_index], data])

# Simulate input_file for automatic basename detection
input_file = "demo_synthetic.tsv"
pd.DataFrame(data, index=species_names).to_csv(input_file, sep="\t")

# Run SID decomposition with automatic file output
I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2,
                             species_names=species_names,
                             input_file=input_file)

# Convert to network format and save
df = sid_to_network_df(I_R, I_S, species_names=species_names, basename="demo_synthetic")

# Build and save network files
build_sid_network(df, output_dir="./sid_output", env_name="demo_synthetic")

# Optional: extract Unique contributions
sid_result_df = pd.read_csv("demo_synthetic_sid_results.tsv", sep="\t")
unique_df = sid_result_df[sid_result_df["Type"] == "Unique"]
print("Unique contributions:")
print(unique_df)
