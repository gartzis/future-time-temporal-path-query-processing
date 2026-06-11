# Data

This folder contains the dataset files and the query-test files used by the experiments.

Expected temporal edge streams:

```text
Data/Datasets/enron.csv
Data/Datasets/email-eu.csv
Data/Datasets/collegemsg.csv
Data/Datasets/bitcoin.csv
```

Expected query-test files:

```text
Data/query_tests/enron.tsv
Data/query_tests/email_eu.tsv
Data/query_tests/collegemsg.tsv
Data/query_tests/bitcoin.tsv
```

Each temporal edge file should contain source, destination, and timestamp columns. Each query-test file should contain source, destination, previous path, future path, and query time fields.
