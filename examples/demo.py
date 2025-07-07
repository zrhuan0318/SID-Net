# SID-Net Example: Decompose information and construct a microbial network

import numpy as np
import pandas as pd
from sidnet import sid_decompose, build_sid_network

# Generate a simple synthetic microbial dataset (3 variables, 1000 time points)
np.random.seed(0)
X = np.random.rand(3, 1000)

# Use the first variable as the target, and the rest as predictors, then perform SID decomposition
Y = np.vstack([X[0], X[1], X[2]])  # target + inputs
I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2)

# Format output as a DataFrame
data = []
for k in I_S:
    if len(k) == 2:
        data.append({
            "source_otu": f"OTU{k[0]}",
            "target_otu": f"OTU{k[1]}",
            "synergy": I_S[k],
            "redundant": (I_R.get((k[0],), 0) + I_R.get((k[1],), 0)) / 2
        })
df = pd.DataFrame(data)

# Build the microbial network
build_sid_network(df, output_dir="./network_output")

# Display the edges (lines.csv)
print("\nEdges (lines.csv):")
print(pd.read_csv("./network_output/lines.csv"))

# Display the nodes (points.csv)
print("\nNodes (points.csv):")
print(pd.read_csv("./network_output/points.csv"))
