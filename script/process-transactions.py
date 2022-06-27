from glob import glob
import gzip
import json
from os import path

from staking.constants import DATA_PATH

if __name__ == "__main__":

    # identify all the files in the data folder that follows the format "transactions-*.jsonl.gz"
    individual_tx_files = sorted(
        glob(path.join(DATA_PATH, "transactions-*.jsonl.gz")), reverse=False
    )
    transactions = []

    # iterate through each file
    for file in individual_tx_files[:2]:
        with gzip.open(file) as f:

            # iterate through each line
            for j, v in enumerate(f):
                transactions.append(json.loads(v)["max_fee"])
                if j % 100 == 0:
                    print(j)
