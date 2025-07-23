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
from typing import Tuple, Dict
import warnings
import pandas as pd  # Newly added for DataFrame export
from . import sid_tools as sid

warnings.filterwarnings("ignore", category=UserWarning)

def sid_decompose(Y: np.ndarray, nbins, max_combs):
    """
    High-dimensional information decomposition (SID):
    Decomposes mutual information between a target and input variables
    into redundant/unique and synergistic contributions.

    Parameters:
    - Y: np.ndarray, shape (n_variables, n_samples)
          First row is the target variable, rest are input variables.
    - nbins: int
          Number of bins for histogram discretization.
    - max_combs: int
          Maximum order of variable combinations to consider.

    Returns:
    - I_R: dict
          Redundant + Unique information contributions.
    - I_S: dict
          Synergistic information contributions.
    - MI: dict
          Mutual information per variable combination.
    """
    Ntot = Y.shape[0]  # total number of variables (1 target + N agents)
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

    return I_R, I_S, MI


def sid_to_network_df(I_R: Dict, I_S: Dict) -> pd.DataFrame:
    """
    Convert SID decomposition results into a DataFrame suitable for SID network construction.

    Parameters:
    - I_R: dict of redundant + unique information
    - I_S: dict of synergistic information

    Returns:
    - pd.DataFrame with columns: source_otu, target_otu, synergy, redundant
    """
    records = []
    for combo in I_S:
        if len(combo) == 2:
            source, target = combo
            synergy = I_S.get(combo, 0.0)
            red_source = I_R.get((source,), 0.0)
            red_target = I_R.get((target,), 0.0)
            redundant = (red_source + red_target) / 2
            records.append({
                "source_otu": f"OTU_{source}",
                "target_otu": f"OTU_{target}",
                "synergy": synergy,
                "redundant": redundant
            })
    return pd.DataFrame.from_records(records)
