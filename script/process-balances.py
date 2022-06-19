import gzip
import json
from os import path

from staking.constants import DATA_PATH

if __name__ == "__main__":

    file_path = path.join(DATA_PATH, "balances.jsonl.gz")
    balances = []
    with gzip.open(file_path) as f:
        for j, v in enumerate(f):
            balances.append(json.loads(v))
            if j % 2_000 == 0:
                print(j)
