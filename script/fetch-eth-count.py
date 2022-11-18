import os
# run `pip install --upgrade google-cloud-bigquery` in the terminal first
from google.cloud import bigquery
from pandas import json_normalize

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "subtle-fulcrum-key.json"
client = bigquery.Client()

# Perform a query.
query_eth = ("""
SELECT Date(block_timestamp) AS block_date, eth_transfer, count(*) AS transaction_count, 
FROM (SELECT block_timestamp, `hash`, value > 0 AS eth_transfer
FROM `bigquery-public-data.crypto_ethereum.transactions` 
)
GROUP BY block_date, eth_transfer
ORDER BY block_date, eth_transfer
""")


if __name__ == "__main__":
    query_job = client.query(query_eth)  # API request
    rows = query_job.result()  # Waits for query to finish


    field_names = [f.name for f in rows.schema]
    # needs to be done in once, otherwise 'Iterator has already started' error
    eth_count = [{
        field: row[field] for field in field_names
    } for row in rows]

    result = json_normalize(eth_count)
