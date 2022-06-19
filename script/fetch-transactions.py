from datetime import datetime
from functools import partial
from glob import glob
import multiprocessing
import logging
import re
from os import path, rename

from staking.constants import DATA_PATH
from staking.utils import save_data

logging.basicConfig(level=logging.INFO)


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
        file_suffix_inscope = [f for f in file_suffixes if greater_than < f < less_than]
        if file_suffix_inscope:
            file_suffix = file_suffixes[0]
            greater_than = file_suffix
        else:
            tx = {}

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

        if not next_query:
            break


break_points = [
    datetime(2019, 1, 1),
    datetime(2020, 4, 1),
    datetime(2020, 7, 1),
    datetime(2020, 10, 1),
    datetime(2021, 1, 1),
    datetime(2021, 4, 1),
    datetime(2021, 7, 1),
    datetime(2021, 10, 1),
    datetime(2022, 1, 1),
    datetime(2022, 4, 1),
    datetime(2022, 7, 1),
]


gt_lt_list = [
    [break_points[i].timestamp(), break_points[i + 1].timestamp()]
    for i in range(0, len(break_points) - 1)
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
            number_iterations=1_000,
            counter_field_url="timestamp",
        ),
        iterable=gt_lt_list,
    )
