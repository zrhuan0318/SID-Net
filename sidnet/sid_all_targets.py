import numpy as np
import pandas as pd
import os
from sid import sid_decompose, sid_to_network_df

def run_sid_all_targets(
    Y: np.ndarray,
    species_names: list,
    output_dir: str,
    basename: str,
    nbins: int = 8,
    max_combs: int = 2
):
    """
    Perform SID decomposition for each OTU as the target.

    Parameters:
        Y: A numpy array (float) of shape OTU x Sample.
        species_names: A list of OTU names (length must equal Y.shape[0]).
        output_dir: Directory path where output files will be saved.
        basename: Prefix for output file names (e.g., environment name).
        nbins: Number of bins for discretization.
        max_combs: Maximum value of synergy dimension K.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for i in range(Y.shape[0]):
        target = Y[i]
        predictors = np.delete(Y, i, axis=0)
        predictor_names = species_names[:i] + species_names[i+1:]
        target_name = species_names[i]

        # Y for SID: first row is target, others are predictors
        Y_sid = np.vstack([target, predictors])

        # --- Key change: pass per-target basename and output_dir so files don't collide ---
        I_R, I_S, MI = sid_decompose(
            Y_sid,
            nbins=nbins,
            max_combs=max_combs,
            species_names=predictor_names,   # first entry is "Target" inside, so predictors map 1..N
            input_file=None,                 # keep for compatibility, not used here
            output_basename=f"{basename}_T{target_name}",
            output_dir=output_dir
        )

        # per-target network df (will also emit "*_df.tsv" if basename is given)
        df_net = sid_to_network_df(
            I_R, I_S,
            species_names=predictor_names,
            basename=os.path.join(output_dir, f"{basename}_T{target_name}")
        )
        df_net["target"] = target_name
        all_results.append(df_net)

    # Combine all networks across targets for this environment
    combined_df = pd.concat(all_results, ignore_index=True)
    final_out_path = os.path.join(output_dir, f"{basename}_all_targets_df.tsv")
    combined_df.to_csv(final_out_path, sep="\t", index=False)
