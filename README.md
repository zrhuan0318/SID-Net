# SID-Net: Synergistic Information Decomposition for Microbial Networks

**SID-Net** is a Python package for analyzing synergistic, redundant, and unique information flows between variables (such as microbial taxa) based on time series or abundance data. It enables researchers to identify cooperative and independent interactions among microbial community members, and construct synergy-driven ecological networks.

## Installation

### Option 1: Install via pip (standard)

```bash
pip install ./
```

This will install the package and required dependencies:
- numpy
- pandas
- torch
- matplotlib

### Option 2: Install via Conda (recommended for reproducibility)

```bash
git clone https://github.com/zrhuan0318/SID-Net.git
cd SID-Net
conda env create -f environment.yml
conda activate sidnet
```

This will:
- Create a new Conda environment `sidnet`
- Install Python 3.9, numpy, pandas, matplotlib, pytorch
- Use `pip install .` to install the SID-Net package from source

## Quick Start

```python
from sidnet import sid_decompose, sid_to_network_df, build_sid_network
import pandas as pd
import numpy as np

# Load OTU table
otu_df = pd.read_csv("OTU_TABLE.txt", sep="\t", index_col=0)
data = otu_df.values.astype(float)

# Prepare input
target_index = 0
Y = np.vstack([data[target_index], data])
species_names = list(otu_df.index)
basename = "OTU_TABLE"

# Run SID decomposition and export results
I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2, species_names=species_names, basename=basename)

# Convert and save network input
df = sid_to_network_df(I_R, I_S, species_names=species_names, basename=basename)

# Build network
build_sid_network(df, output_dir="./sid_output", env_name="otu_sid")
```

> **Note:** If `species_names` and `basename` are provided, `sid_decompose` and `sid_to_network_df` will automatically export result files to the current directory:
>
> - `OTU_TABLE_sid_results.tsv`
> - `OTU_TABLE_unique_contributions.tsv`
> - `OTU_TABLE_df.tsv`

## Folder Structure

```
SID-Net/
└── sidnet/
    ├── sid.py           # SID decomposition logic + DataFrame export
    ├── sid_tools.py     # Information theory tools
    ├── sid_net.py       # Network construction
    └── __init__.py      # Public API
├── environment.yml      # Conda environment
├── setup.py
├── README.md
└── examples/
    ├── demo.py
    └── demo.ipynb
```

## Examples

Run either of the examples:

- `examples/demo.py`: Script-based example
- `examples/demo.ipynb`: Interactive Jupyter Notebook

These demonstrate how to compute SID and build microbial networks from synthetic data.

## License

MIT License

## Author

Developed by Ruihuan Zhang, 2025.
