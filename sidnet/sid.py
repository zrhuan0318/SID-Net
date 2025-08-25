import numpy as np
import pandas as pd
from .sid_tools import compute_all_mi_combinations
from itertools import combinations

def sid_decompose(Y, nbins=5, max_combs=2, species_names=None, input_file=None):
    """
    Perform SID decomposition on input data.

    Parameters
    ----------
    Y : ndarray
        Data matrix of shape (nvars, nsamples).
        First row is the target variable, remaining rows are predictors.
    nbins : int
        Number of bins for discretization.
    max_combs : int
        Maximum size of variable combinations to consider.
    species_names : list
        Names of variables. If provided, the first element is assumed to be the target,
        and the remaining correspond to predictors.
    input_file : str
        Optional, used to generate output file name.

    Returns
    -------
    I_R : dict
        Redundant contributions
    I_S : dict
        Synergistic contributions
    MI : dict
        Mutual information for each variable combination
    """
    if species_names is None:
        # Default names: X1, X2, ... for predictors
        species_names = [f"X{i+1}" for i in range(Y.shape[0] - 1)]
        species_names = ["Target"] + species_names

    # Determine basename for file output
    if input_file:
        import os
        basename = os.path.splitext(os.path.basename(input_file))[0]
    else:
        basename = "sid_output"

    # Run computation
    I_U, I_R, I_S, MI = compute_all_mi_combinations(Y, nbins=nbins, max_combs=max_combs)

    # Write full SID results to a single TSV
    sid_rows = []

    # ---- Unique contributions (skip target!) ----
    for i, val in enumerate(I_U):
        sid_rows.append((species_names[i+1], val, "Unique"))

    # ---- Redundant contributions (excluding singletons) ----
    for k, val in I_R.items():
        if len(k) == 1:
            continue  # skip singletons to avoid duplication
        names = tuple(species_names[i] for i in k)
        sid_rows.append((names, val, "Redundant"))

    # ---- Synergistic contributions ----
    for k, val in I_S.items():
        names = tuple(species_names[i] for i in k)
        sid_rows.append((names, val, "Synergistic"))

    # Write to file
    df = pd.DataFrame(sid_rows, columns=["Feature", "Contribution", "Type"])
    df.to_csv(f"{basename}_sid_results.tsv", sep="\t", index=False)

    return I_R, I_S, MI

def sid_to_network_df(I_R, I_S, species_names=None, basename=None):
    """
    Convert SID decomposition results into a DataFrame representing a network.

    Parameters
    ----------
    I_R : dict
        Redundant contributions
    I_S : dict
        Synergistic contributions
    species_names : list
        Names of variables. The first element corresponds to target, others to predictors.
    basename : str
        Optional file prefix for output.

    Returns
    -------
    df : DataFrame
        Network edges with synergy and redundancy values.
    """
    data = []
    for k in I_S:
        if len(k) == 2:
            s1, s2 = k
            synergy = I_S[k]
            r1 = I_R.get((s1,), 0)
            r2 = I_R.get((s2,), 0)
            avg_redundant = (r1 + r2) / 2
            data.append({
                "source_otu": species_names[s1],
                "target_otu": species_names[s2],
                "synergy": synergy,
                "redundant": avg_redundant
            })

    df = pd.DataFrame(data)
    if basename:
        df.to_csv(f"{basename}_df.tsv", sep="\t", index=False)
    return df
