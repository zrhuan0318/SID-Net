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
from sidnet import sid_decompose, build_sid_network
import numpy as np
import pandas as pd

# Example data: 3 variables, 1000 samples
X = np.random.rand(3, 1000)
Y = np.vstack([X[0], X[1], X[2]])  # Target + inputs

# Perform SID
I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2)

# Build network
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
build_sid_network(df)
```

## Folder Structure

```
SID-Net/
└── sidnet/
    ├── sid.py               # SID decomposition logic
    ├── sid_tools.py     # Information theory tools
    ├── sid_net.py        # Network construction
    └── __init__.py      # Public API
├── environment.yml   # Conda environment
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
