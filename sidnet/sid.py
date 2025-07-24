import numpy as np
import pandas as pd
import itertools
import os
from .sid_tools import compute_all_mi_combinations

def sid_decompose(Y, nbins, max_combs=2, species_names=None, basename=None, input_file=None):
    if basename is None and input_file is not None:
        basename = os.path.splitext(os.path.basename(input_file))[0]

    I_U, I_R, I_S, MI = compute_all_mi_combinations(Y, nbins=nbins, max_combs=max_combs)

    # Write full result table if basename is given
    if basename is not None:
        all_records = []
        for comb, value in I_R.items():
            name = tuple(species_names[i] for i in comb) if species_names else comb
            all_records.append({'Feature': name, 'Contribution': value, 'Type': 'Redundant'})
        for comb, value in I_S.items():
            name = tuple(species_names[i] for i in comb) if species_names else comb
            all_records.append({'Feature': name, 'Contribution': value, 'Type': 'Synergistic'})
        for i, value in enumerate(I_U):
            name = species_names[i] if species_names else f"X{i}"
            all_records.append({'Feature': name, 'Contribution': value, 'Type': 'Unique'})

        df = pd.DataFrame(all_records)
        df.to_csv(f"{basename}_sid_results.tsv", sep='\t', index=False)

    return I_R, I_S, MI

def sid_to_network_df(I_R, I_S, species_names=None, basename=None):
    data = []
    for comb, syn_value in I_S.items():
        if len(comb) == 2:
            s1, s2 = comb
            s1_name = species_names[s1] if species_names else f"X{s1}"
            s2_name = species_names[s2] if species_names else f"X{s2}"
            r1 = I_R.get((s1,), 0)
            r2 = I_R.get((s2,), 0)
            data.append({
                "source_otu": s1_name,
                "target_otu": s2_name,
                "synergy": syn_value,
                "redundant": (r1 + r2) / 2
            })
    df = pd.DataFrame(data)

    if basename is not None:
        df.to_csv(f"{basename}_df.tsv", sep='\t', index=False)

    return df
