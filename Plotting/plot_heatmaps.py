from __future__ import annotations



from pathlib import Path



import matplotlib.pyplot as plt

import pandas as pd





INPUT_FILE = Path("Results/RQ2_optimal_temporal_edge_oracle/heatmap_values.tsv")

OUTPUT_FILE = Path("Results/figures/rq2_heatmap.png")





def main() -> None:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    pivot = df.pivot(index="edge_probability", columns="shortest_path_threshold", values="coverage")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))

    plt.imshow(pivot.values, aspect="auto", origin="lower")

    plt.xticks(range(len(pivot.columns)), pivot.columns)

    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("Shortest-path threshold")

    plt.ylabel("Edge probability")

    plt.colorbar(label="Coverage")

    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300)





if __name__ == "__main__":

    main()

