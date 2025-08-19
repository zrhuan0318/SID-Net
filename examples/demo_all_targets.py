import numpy as np
import pandas as pd
from sidnet import run_sid_all_targets, build_sid_network

# Generate synthetic data
np.random.seed(42)
species_names = [f"Species_{i}" for i in range(5)]
data = np.random.rand(5, 1000)

# Run SID across all OTUs as targets
run_sid_all_targets(
    Y=data,
    species_names=species_names,
    output_dir="./",
    basename="demo",
    nbins=5,
    max_combs=2
)

# Load the resulting synergy network across all targets
df_all = pd.read_csv("demo_all_targets_df.tsv", sep="\t")

# Construct a global network from all synergy edges
df_global = df_all[["source", "target", "synergy"]].copy()

# Optionally, filter top synergy edges
# df_global = df_global.sort_values("synergy", ascending=False).head(100)

# Build the network
build_sid_network(df_global)
