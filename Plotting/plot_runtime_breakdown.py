from __future__ import annotations



from pathlib import Path



import matplotlib.pyplot as plt

import pandas as pd





INPUT_FILE = Path("Results/runtime_breakdown.tsv")

OUTPUT_FILE = Path("Results/figures/runtime_breakdown.png")





def main() -> None:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    ax = df.plot(kind="bar", x="step", y="seconds", legend=False)

    ax.set_ylabel("Average time averaged over 10 queries (seconds)")

    plt.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUTPUT_FILE, dpi=300)





if __name__ == "__main__":

    main()

