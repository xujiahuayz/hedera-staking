import json
from os import path
from staking.utils import fetch_data
from staking.constants import DATA_PATH


reward_data = fetch_data("limit=100", "order=asc", "account.id=gt:0.0.0", q="balances")

with open(path.join(DATA_PATH, "balances.json"), "w") as f:
    json.dump(reward_data, f, indent=4)
