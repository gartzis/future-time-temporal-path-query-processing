from __future__ import annotations



from pathlib import Path

import sys
import os



BASE_DIR = Path(__file__).resolve().parents[1]
os.chdir(BASE_DIR)

if str(BASE_DIR) not in sys.path:

    sys.path.insert(0, str(BASE_DIR))



import main as runner





DATASET_NAME = "enron"

DATA_ROOT = "Data/Datasets"

QUERY_TEST_ROOT = "Data/query_tests"

OUTPUT_ROOT = "Results/RQ5_runtime_cache"

ORACLE_NAME = "n2vlp_static"

NUM_QUERIES = 100

NUM_RUNS = 10

TOP_K = 10

NUM_LANDMARKS = 5

EDGE_THRESHOLD = 0.5

PATH_EXIST_THRESHOLD = 0.0

SHORTEST_PATH_THRESHOLD = 0.0

USE_CACHE = True

QUERY_SAMPLE_SEED = 42






def main() -> None:

    runner.DATASET_NAME = DATASET_NAME

    runner.DATA_ROOT = DATA_ROOT

    runner.QUERY_TEST_ROOT = QUERY_TEST_ROOT

    runner.OUTPUT_ROOT = OUTPUT_ROOT

    runner.ORACLE_NAME = ORACLE_NAME

    runner.NUM_QUERIES = NUM_QUERIES

    runner.NUM_RUNS = NUM_RUNS

    runner.TOP_K = TOP_K

    runner.NUM_LANDMARKS = NUM_LANDMARKS

    runner.EDGE_THRESHOLD = EDGE_THRESHOLD

    runner.PATH_EXIST_THRESHOLD = PATH_EXIST_THRESHOLD

    runner.SHORTEST_PATH_THRESHOLD = SHORTEST_PATH_THRESHOLD

    runner.USE_CACHE = USE_CACHE

    runner.QUERY_SAMPLE_SEED = QUERY_SAMPLE_SEED


    runner.main()





if __name__ == "__main__":

    main()

