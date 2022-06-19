from genericpath import exists
from glob import glob
import json
import time
import requests
import logging
import gzip
import re
from os import path, remove, rename

from staking.constants import DATA_PATH

logging.basicConfig(level=logging.INFO)


def save_data(
    file_name: str,
    q: str = "transactions",
    suburl: str = "api/v1",
    rooturl: str = "https://mainnet-public.mirrornode.hedera.com",
    n_pages: int = 1_000,
    **kwargs,
):

    default_file = path.join(DATA_PATH, f"{file_name}.jsonl.gz")

    breakpoint_file = path.join(DATA_PATH, f"{file_name}-breakpoint.txt")

    if exists(breakpoint_file):
        with open(breakpoint_file) as text_file:
            query = text_file.readline()
        remove(breakpoint_file)
    else:
        args = []
        for key, value in kwargs.items():
            args.append(f"{key}={value}")
        query = f"/{suburl}/{q}?{'&'.join(args)}"

    with gzip.open(default_file, "at") as f:

        for j in range(n_pages):

            if j % 100 == 0:
                logging.info(f"page {j} ==== {query}")

            request_url = rooturl + query

            try:
                r = requests.get(request_url)
                response = json.loads(r.text)
            except:
                print(f"error at ==== {query}, retrying")
                time.sleep(100)
                try:
                    r = requests.get(request_url)
                    response = json.loads(r.text)
                except:
                    print(f"error at ==== {query}, retry failed")
                    with open(breakpoint_file, "w") as text_file:
                        text_file.write(query)
                    break

            for tx in response[q]:
                f.write(json.dumps(tx) + "\n")

            query = response["links"]["next"]

            # if query is None or ''
            if not query:
                break

    return query


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

        # log message every 100 queries
        if i % 100 == 0:
            logging.info(f"No.{i}===={query}")

    return response_list
