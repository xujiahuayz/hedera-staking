import gzip
import json
from glob import glob
from os import path
import datetime

individual_tx_files = sorted(
    glob(path.join("./data", "transactions*.jsonl.gz")), reverse=False
)

numfile = len(individual_tx_files)
print(numfile)

begin = datetime.date(2019, 9, 13)
# end=datetime.date(2022,6,30)


d = begin
# delta=datetime.timedelta(days=1)


# transactions = []

Tx_num = {}

Node_fee_T = {}
node_fee = {}

Treasury_fee = {}

# iterate through each file
for file in individual_tx_files:
    with gzip.open(file) as f:

        # iterate through each line
        for j, v in enumerate(f):

            Tx_time = json.loads(v)["consensus_timestamp"]
            date = datetime.datetime.utcfromtimestamp(float(Tx_time)).strftime(
                "%Y-%m-%d"
            )
            if d != date:
                Node_fee_T[date] = node_fee.copy()
                d = date

            Tx_num[date] = Tx_num.get(date, 0) + 1
            # charged_fee[date] = charged_fee.get(date,0) + json.loads(v)['charged_tx_fee']

            charged_fee_tx = json.loads(v)["charged_tx_fee"]
            node = json.loads(v)["node"]
            transfers = json.loads(v)["transfers"]

            for i, n in enumerate(transfers):
                if n["account"] == node:
                    node_fee[node] = node_fee.get(node, 0) + n["amount"]
                    treasury_fee_tx = charged_fee_tx - n["amount"]

                    Treasury_fee[date] = Treasury_fee.get(date, 0) + treasury_fee_tx

                    break

            # transactions.append(json.loads(v)['consensus_timestamp'])
    numfile -= 1
    print("residue file", numfile)
