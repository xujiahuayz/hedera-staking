from os import path
import requests
from datetime import datetime
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from staking.settings import PROJECT_ROOT

# https://etherscan.io/chart/tx?output=csv
# https://arbiscan.io/chart/tx?output=csv
# https://polygonscan.com/chart/tx?output=csv
# https://bscscan.com/chart/tx?output=csv


def get_exchange_volume(chain: str = "eth") -> list[dict]:
    """
    fetch historical volume data
    """
    csv_path = path.join(PROJECT_ROOT, "data", f"{chain}-TxGrowth.csv")
    return pd.read_csv(csv_path).to_dict("records")


def get_day_volume(volumes: list[dict]) -> tuple[list, list]:
    """
    volume data by age in days from age 0
    """

    volume_list = [float(w["Value"]) for w in volumes]
    time_list = [w["UnixTimeStamp"] for w in volumes]
    start_time = time_list[0]
    day_list = [(w - start_time) / 24 / 3600 for w in time_list]

    return day_list, volume_list


def plot_exchange_volume(exchange: str, color: str) -> None:
    exchange_data = get_exchange_volume(chain=exchange)
    x, y = get_day_volume(exchange_data)
    plot_volume(x, y, exchange, color)


def plot_volume(x: list[float], y: list[float], chain: str, color: str) -> None:
    log_x = [np.log(w + 1) for w in x]
    log_y = [np.log(w + 1) for w in y]
    coefficients = np.polyfit(log_x, log_y, 1)
    fitted_x = range(round(max(x)))
    fitted_y = [coefficients[0] * np.log(w + 1) + coefficients[1] for w in fitted_x]

    plt.plot(y, color=color, linestyle="dotted", alpha=0.8)
    plt.plot(
        [np.exp(w) - 1 for w in fitted_y],
        linewidth=2,
        color=color,
        label=f"{chain} ({coefficients})",
    )


if __name__ == "__main__":

    plot_exchange_volume("eth", "blue")
    plot_exchange_volume("arbitrum", "orange")
    plot_exchange_volume("polygon", "red")
    plot_exchange_volume("bnb", "cyan")

    plt.yscale("log")
    plt.xlabel("$age$ (day)")
    plt.ylabel("$TXcount$ ")
    plt.legend()
    plt.title("$\\log(TXcount + 1) = a \\times \\log(age + 1) + b$")
