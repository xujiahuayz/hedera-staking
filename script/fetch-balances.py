# import json
# from os import path
from staking.utils import fetch_data, save_data

# from staking.constants import DATA_PATH

if __name__ == "__main__":
    save_data(
        file_name="balances", q="balances", n_pages=99_999, limit=100, order="asc"
    )

# balances_data = fetch_data(
#     "limit=100", "order=asc", "account.id=gt:0.0.0", q="balances"
# )

# with open(path.join(DATA_PATH, "balances.json"), "w") as f:
#     json.dump(balances_data, f, indent=4)
