import json
from os import path
from staking.utils import fetch_data
from staking.constants import DATA_PATH

if __name__ == "__main__":
    reward_tx_data = fetch_data(
        "limit=100", "order=asc", "account.id=0.0.800", q="transactions"
    )

    with open(path.join(DATA_PATH, "reward_transactions.json"), "w") as f:
        json.dump(reward_tx_data, f, indent=4)
