from __future__ import annotations



from pathlib import Path



import pandas as pd





INPUT_FILE = Path("Results/oracle_temporal_results_summary.tsv")

OUTPUT_FILE = Path("Results/tables/oracle_temporal_results_table.tex")





def main() -> None:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(df.to_latex(index=False), encoding="utf-8")





if __name__ == "__main__":

    main()

