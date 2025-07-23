
# sid.py
"""
SID: Synergistic Information Decomposition
This module provides high-dimensional information decomposition from a target variable
and multiple input variables into unique, redundant, and synergistic components.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from itertools import combinations as icmb
from itertools import chain as ichain
from typing import Tuple, Dict, List, Optional
import warnings
import pandas as pd
import os
from . import sid_tools as sid

warnings.filterwarnings("ignore", category=UserWarning)

def sid_decompose(Y: np.ndarray, nbins: int, max_combs: int, species_names: Optional[List[str]] = None, basename: Optional[str] = None):
    """
    Perform information decomposition and optionally save the results.

    Parameters:
    - Y: np.ndarray, shape (n_variables, n_samples)
          First row is the target variable, rest are input variables.
    - nbins: int
          Number of bins for histogram discretization.
    - max_combs: int
          Maximum order of variable combinations to consider.
    - species_names: list of str (optional)
          Names corresponding to each variable (excluding target).
    - basename: str (optional)
          Base name for output files, without extension.

    Returns:
    - I_R: dict
          Redundant + Unique information contributions.
    - I_S: dict
          Synergistic information contributions.
    - MI: dict
          Mutual information per variable combination.
    """
    Ntot = Y.shape[0]
    tot_inds = range(1, Ntot)
    p_target = sid.myhistogram(Y[0, :].reshape(-1, 1), nbins)

    combs, Is = [], {}
    for r in range(1, max_combs + 1):
        for combo in itertools.combinations(tot_inds, r):
            combs.append(combo)
            selected_vars = (0,) + combo
            hist = sid.myhistogram(Y[selected_vars, :].T, nbins)
            p_joint = hist
            p_combo = hist.sum(axis=0) if len(combo) == 1 else hist.sum(axis=0)
            p_target_given_combo = p_joint / (p_combo + 1e-14)
            p_target_expand = p_target.reshape(-1, *(1,) * len(combo))
            s_info = (p_target_given_combo * (sid.mylog(p_target_given_combo) - sid.mylog(p_target_expand))).sum(
                axis=tuple(range(1, len(selected_vars)))
            )
            Is[combo] = s_info

    MI = {k: (Is[k] * p_target.squeeze()).sum() for k in Is.keys()}
    I_R = {cc: 0 for cc in combs if len(cc) == 1 or len(cc) == 2}
    I_S = {cc: 0 for cc in combs if len(cc) >= 2}

    nbins_target = p_target.shape[0]
    for t in range(nbins_target):
        I1 = np.array([ii[t] for ii in Is.values()])
        i1 = np.argsort(I1)
        lab = [combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])
        I1 = I1[i1]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l+1)[0]
            Il1max = I1[lens == l].max()
            inds_ = inds_l2[I1[inds_l2] < Il1max]
            I1[inds_] = 0
        i1 = np.argsort(I1)
        lab = [lab[i_] for i_ in i1]
        Di = np.diff(I1[i1], prepend=0.)
        red_vars = list(tot_inds)
        for i_, ll in enumerate(lab):
            info = Di[i_] * p_target.squeeze()[t]
            if len(ll) == 1:
                I_R[ll] += info
                red_vars.remove(ll[0])
            else:
                I_S[ll] += info

    # Save results if basename is provided
    if basename is not None and species_names is not None:
        rows_all = []
        rows_unique = []
        for k, v in I_R.items():
            names = tuple(species_names[i - 1] for i in k)
            label = f"({', '.join(names)})"
            if len(k) == 1:
                rows_unique.append([names[0], v])
                rows_all.append([label, v, "Unique"])
            else:
                rows_all.append([label, v, "Redundant"])
        for k, v in I_S.items():
            names = tuple(species_names[i - 1] for i in k)
            label = f"({', '.join(names)})"
            rows_all.append([label, v, "Synergistic"])

        df_all = pd.DataFrame(rows_all, columns=["Feature", "Contribution", "Type"])
        df_unique = pd.DataFrame(rows_unique, columns=["Species", "Unique_Contribution"])
        df_all.to_csv(f"{basename}_sid_results.tsv", sep="\t", index=False)
        df_unique.to_csv(f"{basename}_unique_contributions.tsv", sep="\t", index=False)

    return I_R, I_S, MI


def sid_to_network_df(I_R: Dict, I_S: Dict, species_names: Optional[List[str]] = None, basename: Optional[str] = None) -> pd.DataFrame:
    """
    Convert SID results into DataFrame and optionally save to file.

    Parameters:
    - I_R: dict of redundant + unique information
    - I_S: dict of synergistic information
    - species_names: list of OTU names
    - basename: base name for file export

    Returns:
    - pd.DataFrame with source_otu, target_otu, synergy, redundant
    """
    records = []
    for combo in I_S:
        if len(combo) == 2:
            i, j = combo
            synergy = I_S.get(combo, 0.0)
            red_i = I_R.get((i,), 0.0)
            red_j = I_R.get((j,), 0.0)
            redundant = (red_i + red_j) / 2
            source = species_names[i - 1] if species_names else f"OTU_{i}"
            target = species_names[j - 1] if species_names else f"OTU_{j}"
            records.append({
                "source_otu": source,
                "target_otu": target,
                "synergy": synergy,
                "redundant": redundant
            })
    df = pd.DataFrame.from_records(records)
    if basename is not None:
        df.to_csv(f"{basename}_df.tsv", sep="\t", index=False)
    return df
