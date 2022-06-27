from genericpath import exists
from glob import glob
import json
import re
import time
from typing import Optional
import requests
import logging
import gzip
from os import path, remove, rename

from staking.constants import DATA_PATH

logging.basicConfig(level=logging.INFO)


def save_data(
    file_name: str,
    q: str = "transactions",
    suburl: str = "api/v1",
    rooturl: str = "https://mainnet-public.mirrornode.hedera.com",
    n_pages: int = 1_000,
    query: Optional[str] = None,
    **kwargs,
):

    default_file = path.join(DATA_PATH, f"{file_name}.jsonl.gz")

    breakpoint_file = path.join(DATA_PATH, f"{file_name}-breakpoint.txt")

    if exists(breakpoint_file):
        with open(breakpoint_file) as text_file:
            query = text_file.readline()
        remove(breakpoint_file)
    else:
        if not query:
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
            # logging.info(f"page {j} ==== {query}")

            # if query is None or ''
            if not query:
                break

    return query


def fetch_save_data(
    gt_lt=[1500000000, 2000000000],
    limit: int = 100,
    file_name: str = "transactions",
    q: str = "transactions",
    suburl: str = "api/v1",
    rooturl: str = "https://mainnet-public.mirrornode.hedera.com",
    number_iterations: int = 2,
    counter_field_url: str = "timestamp",
):
    """
    get data from hedera mirror node
    run save_data under the hood with ascending ordering
    """

    individual_tx_files = sorted(
        glob(path.join(DATA_PATH, f"{file_name}-*.jsonl.gz")), reverse=True
    )

    greater_than = gt_lt[0]
    less_than = gt_lt[1]

    default_file_name = file_name + str(greater_than) + "temp"
    default_file = path.join(DATA_PATH, f"{default_file_name}.jsonl.gz")

    if individual_tx_files:
        file_suffixes = [
            float(re.split(pattern="-", string=f)[-2]) for f in individual_tx_files
        ]
        file_suffix_inscope = [
            f for f in file_suffixes if float(greater_than) < f < float(less_than)
        ]
        # print(file_suffix_inscope[:10])
        if file_suffix_inscope:
            greater_than = file_suffix_inscope[0]

    query = f"/{suburl}/{q}?limit={limit}&order=asc&{counter_field_url}=lt:{less_than}&{counter_field_url}=gt:{greater_than}"

    logging.info(f"between {greater_than} and {less_than} start querying {query}")

    for i in range(number_iterations):

        next_query = save_data(
            file_name=default_file_name,
            q=q,
            suburl=suburl,
            rooturl=rooturl,
            n_pages=1_000,
            order="asc",
            limit=limit,
            query=query,
        )

        if next_query:
            query = next_query
            file_suffix = next_query.split("gt:")[1]
            rename(
                default_file,
                path.join(DATA_PATH, f"{file_name}-{file_suffix}-.jsonl.gz"),
            )
            logging.info(
                f"between {greater_than} and {less_than} No.{i}===={file_suffix}"
            )
        else:
            break


# def fetch_data(
#     *args,
#     q: str = "balances",
#     suburl: str = "api/v1",
#     rooturl: str = "https://mainnet-public.mirrornode.hedera.com",
# ) -> list:
#     """
#     get data from hedera mirror node
#     fetch balances data by default
#     """

#     query = f"/{suburl}/{q}?{'&'.join(args)}"
#     response_list = []

#     for i in range(100_000):
#         request_url = rooturl + query
#         r = requests.get(request_url)

#         response = json.loads(r.text)

#         response_list.extend(response[q])
#         query = response["links"]["next"]

#         if query is None:
#             break

#         # log message every 100 queries
#         if i % 100 == 0:
#             logging.info(f"No.{i}===={query}")

#     return response_list
