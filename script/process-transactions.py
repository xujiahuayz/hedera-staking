from glob import glob
import gzip
import json
from os import path

from staking.constants import DATA_PATH

if __name__ == "__main__":

    individual_tx_files = sorted(
        glob(path.join(DATA_PATH, "transactions-*.jsonl.gz")), reverse=True
    )
    transactions = []

    for file in individual_tx_files:
        with gzip.open(file) as f:
            for j, v in enumerate(f):
                transactions.append(json.loads(v))
                if j % 100 == 0:
                    print(j)


# INFO:root:No.0====/api/v1/transactions?limit=100&order=asc&timestamp=gt:1568412964.375367001
