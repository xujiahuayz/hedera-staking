import json
from os import path
from staking.utils import fetch_data
from staking.constants import DATA_PATH


balances_data = fetch_data(
    "limit=100", "order=asc", "account.id=0.0.800", q="transactions"
)

with open(path.join(DATA_PATH, "reward_transactions.json"), "w") as f:
    json.dump(balances_data, f, indent=4)
