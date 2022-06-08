import json
import requests
import logging

logging.basicConfig(level=logging.INFO)


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

    for i in range(5_000):
        request_url = rooturl + query
        r = requests.get(request_url)

        response = json.loads(r.text)

        response_list.extend(response[q])
        query = response["links"]["next"]

        if query is None:
            break

        logging.info(f"No.{i}===={query}")

    return response_list
