# sid_network_tools.py
import pandas as pd
import os

def build_sid_network(input_df: pd.DataFrame, output_dir: str = "./network_output", env_name: str = "network") -> None:
    os.makedirs(output_dir, exist_ok=True)

    filtered_df = input_df[input_df["synergy"] > 0].copy()

    lines_df = filtered_df[["source_otu", "target_otu", "synergy"]].copy()
    lines_df.columns = ["source", "target", "weight"]
    lines_df.to_csv(os.path.join(output_dir, "lines.csv"), index=False)

    otus = pd.Series(pd.concat([filtered_df["source_otu"], filtered_df["target_otu"]])).unique()
    node_df = pd.DataFrame(otus, columns=["id"])

    redundant_dict = {}
    for col in ["source_otu", "target_otu"]:
        grouped = filtered_df.groupby(col)["redundant"].mean()
        redundant_dict.update(grouped)

    node_df["redundant"] = node_df["id"].map(redundant_dict).fillna(0)
    node_df.to_csv(os.path.join(output_dir, "points.csv"), index=False)
