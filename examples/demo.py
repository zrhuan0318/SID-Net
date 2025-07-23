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

# Set output basename
basename = "demo_synthetic"

# Run SID decomposition
I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2, species_names=species_names, basename=basename)

# Convert to network format and save
df = sid_to_network_df(I_R, I_S, species_names=species_names, basename=basename)

# Build and save network files
build_sid_network(df, output_dir="./sid_output", env_name="demo_synthetic")
