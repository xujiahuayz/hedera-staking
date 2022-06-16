from glob import glob
import json
import requests
import logging
import gzip
import re
from os import path, rename

from staking.constants import DATA_PATH

logging.basicConfig(level=logging.INFO)


def fetch_save_data(
    limit: int = 100,
    order: str = "asc",
    file_name: str = "transactions",
    q: str = "transactions",
    suburl: str = "api/v1",
    rooturl: str = "https://mainnet-public.mirrornode.hedera.com",
    number_iterations: int = 2,
    counter_field_url: str = "timestamp",
):
    """
    get data from hedera mirror node
    """

    query = f"/{suburl}/{q}?limit={limit}&order={order}"

    individual_tx_files = sorted(
        glob(path.join(DATA_PATH, f"{file_name}-*.jsonl.gz")), reverse=True
    )

    if len(individual_tx_files) > 0:
        head, tail = path.split(individual_tx_files[0])
        file_suffix = re.split(pattern="-", string=tail)[1]
        query = f'{query}&{counter_field_url}={"gt" if order == "asc" else "lt"}:{file_suffix}'
    else:
        tx = {}

    logging.info(f"start querying {query}")
    default_file = path.join(DATA_PATH, f"{file_name}.jsonl.gz")

    for i in range(number_iterations):

        with gzip.open(default_file, "wt") as f:

            for _ in range(1_000):
                request_url = rooturl + query
                r = requests.get(request_url)

                response = json.loads(r.text)

                for tx in response[q]:
                    f.write(json.dumps(tx) + "\n")

                query = response["links"]["next"]

                if query is None:
                    break

            if query is not None:
                file_suffix = query.split(":")[1]
                rename(
                    default_file,
                    path.join(DATA_PATH, f"{file_name}-{file_suffix}-.jsonl.gz"),
                )
                logging.info(f"% No.{i}===={file_suffix}")

        if query is None:
            break


def fetch_data(
    *args,
    q: str = "balances",
    suburl: str = "api/v1",
    rooturl: str = "https://mainnet-public.mirrornode.hedera.com",
) -> list:
    """
    get data from hedera mirror node
    fetch balances data by default
    """

    query = f"/{suburl}/{q}?{'&'.join(args)}"
    response_list = []

    for i in range(100_000):
        request_url = rooturl + query
        r = requests.get(request_url)

        response = json.loads(r.text)

        response_list.extend(response[q])
        query = response["links"]["next"]

        if query is None:
            break

        # log message every 10 queries
        if i % 10 == 0:
            logging.info(f"No.{i}===={query}")

    return response_list
