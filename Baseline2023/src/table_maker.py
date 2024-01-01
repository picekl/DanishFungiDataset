import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import wandb


def get_results_df(tags: set) -> pd.DataFrame:
    api = wandb.Api()
    entity, project = 'zcu_cv', "DanishFungi2023"

    filtered_tags = {
        "tags": {"$in": ["Production"]},  # T
    }

    runs = api.runs(entity + "/" + project, filters=filtered_tags)
    results_map = {}
    for run in runs:
        # Filter not finished
        state = run.__getattr__("state")
        if state.lower() != "finished":
            continue
        # Has correct tags
        run_tags = run.__getattr__("tags")
        if any([tag not in run_tags for tag in tags]):
            continue

        history = run.history()

        metrics = history[["Val. F1", "Val. Accuracy", "Val. Recall@3"]]
        final_metrics = metrics[~metrics.isnull().any(axis=1)].iloc[-1]

        results_map[run.name] = final_metrics

    results_df = pd.DataFrame.from_dict(results_map, columns=["Val. F1", "Val. Accuracy", "Val. Recall@3"], orient="index")

    results_df *= 100

    results_df = results_df.round(decimals=2)
    return results_df


def main():
    output_path = "../output/Results.txt"
    tags = {"384x384", "Production"}
    results_df = get_results_df(tags)
    results_df = results_df.sort_values(["Val. F1", "Val. Accuracy", "Val. Recall@3"])
    result_message = f"\nDF20M - 384x384 - Production\n{results_df.to_markdown()}\n"
    print(result_message)

    with open(output_path, "a") as f:
        f.write(result_message)


if __name__ == "__main__":
    main()
