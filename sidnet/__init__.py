"""
sidnet: Synergistic Information Decomposition for microbial networks

This package provides tools to analyze synergistic, redundant, and unique information transfer between variables in microbial communities.

Core functionality:
- sid_decompose: Perform information decomposition between variables
- build_sid_network: Build a synergy-based microbial network from SID output

Example usage:

    from sidnet import sid_decompose, build_sid_network

    # Run SID on data matrix Y (target + input variables)
    I_R, I_S, MI = sid_decompose(Y, nbins=5, max_combs=2)

    # Convert results to DataFrame and build network
    df = ...
    build_sid_network(df)

Author: Ruihuan Zhang
License: MIT
"""

__version__ = "0.1"

from .sid import sid_decompose
from .sid_net import build_sid_network
