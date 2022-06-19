from functools import partial
from genericpath import exists
from glob import glob
import json
import multiprocessing
import requests
import logging
import gzip
import re
from os import path, remove, rename

from staking.constants import DATA_PATH
from staking.utils import fetch_data, save_data

logging.basicConfig(level=logging.INFO)


def fetch_save_data(
    gt_lt: list[float] = [1500000000, 2000000000],
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
    """

    individual_tx_files = sorted(
        glob(path.join(DATA_PATH, f"{file_name}-*.jsonl.gz")), reverse=True
    )

    greater_than = gt_lt[0]
    less_than = gt_lt[1]

    if individual_tx_files:
        file_suffixes = [
            float(re.split(pattern="-", string=f)[-2]) for f in individual_tx_files
        ]
        file_suffix_inscope = [f for f in file_suffixes if greater_than < f < less_than]
        if file_suffix_inscope:
            file_suffix = file_suffixes[0]
            greater_than = file_suffix
        else:
            tx = {}

    query = f"/{suburl}/{q}?limit={limit}&order=asc&{counter_field_url}=lt:{less_than}&{counter_field_url}=gt:{greater_than}"

    logging.info(f"start querying {query}")

    default_file_name = file_name + str(greater_than)
    default_file = path.join(DATA_PATH, f"{default_file_name}.jsonl.gz")

    for i in range(number_iterations):

        query = save_data(
            file_name=default_file_name,
            q=q,
            suburl=suburl,
            rooturl=rooturl,
            n_pages=1_000,
            order="asc",
            limit=limit,
        )

        if query:
            file_suffix = query.split(":")[1]
            rename(
                default_file,
                path.join(DATA_PATH, f"{file_name}-{file_suffix}-.jsonl.gz"),
            )
            logging.info(f"% No.{i}===={file_suffix}")

        if not query:
            break


# 01.01 of 2019, 2020, 2021, 2022, 2023
break_points = [1546300800.0, 1577836800.0, 1609459200.0, 1640995200.0, 1672531200.0]

gt_lt_list = [
    [break_points[i], break_points[i + 1]] for i in range(0, len(break_points) - 1)
]

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    pool.map(
        func=partial(
            fetch_save_data,
            limit=100,
            file_name="transactions",
            q="transactions",
            suburl="api/v1",
            rooturl="https://mainnet-public.mirrornode.hedera.com",
            number_iterations=2,
            counter_field_url="timestamp",
        ),
        iterable=gt_lt_list,
    )
