# SID-Net: Synergistic Information Decomposition for Microbial Networks

**SID-Net** is a Python package for analyzing synergistic, redundant, and unique information flows between variables (such as microbial taxa) based on time series or abundance data. It enables researchers to identify cooperative and independent interactions among microbial community members, and construct synergy-driven ecological networks.

## Installation

### Option 1: Install via pip (standard)

```bash
git clone https://github.com/zrhuan0318/SID-Net.git
cd SID-Net
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
import numpy as np

# Example data: 3 variables, 1000 samples
X = np.random.rand(3, 1000)
Y = np.vstack([X[0], X[1], X[2]])  # Target + inputs

# Perform SID decomposition
I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2)

# Convert results to network DataFrame
df = sid_to_network_df(I_R, I_S)

# Build and export network files
build_sid_network(df)
```

> A helper function `sid_to_network_df` is provided to convert SID decomposition results (`I_R`, `I_S`) into the required input format for `build_sid_network()`.

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
