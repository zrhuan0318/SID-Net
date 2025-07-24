# SID-Net: Synergistic Information Decomposition for Microbial Networks

**SID-Net** is a Python package for analyzing synergistic, redundant, and unique information flows between variables (such as microbial taxa) based on time series or abundance data. It enables researchers to identify cooperative and independent interactions among microbial community members, and construct synergy-driven ecological networks.

## Installation

### Option 1: Install via pip

```bash
pip install ./
```

Dependencies:
- numpy
- pandas
- torch
- matplotlib

### Option 2: Install via Conda

```bash
git clone https://github.com/zrhuan0318/SID-Net.git
cd SID-Net
conda env create -f environment.yml
conda activate sidnet
```

## Quick Start (with synthetic data)

```python
import numpy as np
from sidnet import sid_decompose, sid_to_network_df, build_sid_network

# Generate synthetic data
np.random.seed(42)
species_names = [f"Species_{i}" for i in range(5)]
data = np.random.rand(5, 1000)
Y = np.vstack([data[0], data])
input_file = "demo_synthetic.tsv"

# Run SID
I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2,
                             species_names=species_names,
                             input_file=input_file)
df = sid_to_network_df(I_R, I_S, species_names=species_names, basename="demo_synthetic")
build_sid_network(df)
```

> If `input_file` is provided, SID-Net will automatically derive a `basename` for file outputs:
>
> - `demo_synthetic_sid_results.tsv`
> - `demo_synthetic_df.tsv`

## Folder Structure

```
SID-Net/
└── sidnet/
    ├── sid.py           # SID decomposition logic
    ├── sid_tools.py     # Info theory utilities
    ├── sid_net.py       # Network builder
    └── __init__.py
├── examples/
│   ├── demo.py
│   └── demo.ipynb
├── environment.yml
├── setup.py
└── README.md
```

## License

MIT License

## Author

Developed by Ruihuan Zhang, 2025.
