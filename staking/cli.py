from argparse import ArgumentParser
import logging


from staking.utils import fetch_save_data


parser = ArgumentParser(prog="hedera-staking", description="CLI for hedera staking")
parser.add_argument(
    "--debug", default=False, action="store_true", help="Enables debug logging"
)

subparsers = parser.add_subparsers(dest="command")

fetch_transactions_parser = subparsers.add_parser(
    name="fetch-transactions", help="Fetch historical data"
)


fetch_transactions_parser.add_argument(
    "--timestamp-gt",
    required=True,
    help="timestamp greater than, starting point to fetch",
)

fetch_transactions_parser.add_argument(
    "--timestamp-lt", required=True, help="timestamp less than, ending point to fetch"
)


def run_fetch_transactions(args):
    greater_than = args["timestamp_gt"]
    less_than = args["timestamp_lt"]
    fetch_save_data(
        gt_lt=[greater_than, less_than],
        limit=100,
        file_name="transactions",
        q="transactions",
        suburl="api/v1",
        rooturl="https://mainnet-public.mirrornode.hedera.com",
        number_iterations=1_000,
        counter_field_url="timestamp",
    )


def run():
    logging_level = logging.INFO
    args = vars(parser.parse_args())
    if args.pop("debug", False):
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level)
    command = args.pop("command", None)
    if not command:
        parser.error("no command given")
    func_name = "run_{0}".format(command.replace("-", "_"))
    func = globals()[func_name]
    func(args)


# run only if called from command line
if __name__ == "__main__":
    run()
