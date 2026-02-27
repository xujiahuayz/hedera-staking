# %%
"""Plot Hedera metric comparisons from `data/hedera_comparison_sep_feb.csv`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "hedera_comparison_sep_feb.csv"
)
DATE_A = "20250910"
DATE_B = "20250221"
DATE_A_LABEL = "2025-09-10"
DATE_B_LABEL = "2025-02-21"
TOTAL_ACCOUNTS = {DATE_A: 4_444_206, DATE_B: 4_585_189}


def _parse_numeric(value: str) -> float:
    """Parse CSV numeric fields into floats."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text in {"", "-", "--"}:
        return np.nan

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    text = text.replace(",", "").strip()
    try:
        number = float(text)
    except ValueError:
        return np.nan

    return -number if negative else number


def _add_date_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Attach parsed date columns using a common prefix."""
    result = df.copy()
    result[f"{prefix}_{DATE_A}"] = result[DATE_A].apply(_parse_numeric)
    result[f"{prefix}_{DATE_B}"] = result[DATE_B].apply(_parse_numeric)
    return result


def prepare_cdf_data(df_percentiles: pd.DataFrame) -> pd.DataFrame:
    """Transform percentile rows into CDF-ready data."""
    balance_df = df_percentiles[
        df_percentiles["metric"].str.contains("_balance")
    ].copy()
    balance_df["percentile"] = (
        balance_df["metric"].str.extract(r"p(\d+(?:\.\d+)?)").astype(float)
    )
    balance_df = _add_date_columns(balance_df, "balance")
    return (
        balance_df[["percentile", f"balance_{DATE_A}", f"balance_{DATE_B}"]]
        .sort_values("percentile")
        .reset_index(drop=True)
    )


def prepare_range_data(
    df_ranges: pd.DataFrame, include_zero_to_one: bool
) -> pd.DataFrame:
    """Transform range rows into bar-plot-ready data."""
    mask = df_ranges["metric"].str.contains("range.")
    if not include_zero_to_one:
        mask &= ~df_ranges["metric"].str.contains("0-1")

    range_df = df_ranges[mask].copy()
    range_df["range_label"] = (
        range_df["metric"].str.extract(r"range\.(.+)\s+HBAR").fillna(range_df["metric"])
    )
    range_df = _add_date_columns(range_df, "accounts")
    return range_df[
        ["range_label", f"accounts_{DATE_A}", f"accounts_{DATE_B}"]
    ].reset_index(drop=True)


def _plot_grouped_bars(
    ax: plt.Axes,
    x_labels: pd.Series,
    values_a: pd.Series,
    values_b: pd.Series,
    title: str,
    ylabel: str,
) -> plt.Axes:
    """Render a 2-series grouped bar chart."""
    x = np.arange(len(x_labels))
    width = 0.35
    ax.bar(x - width / 2, values_a, width, label=DATE_A_LABEL, alpha=0.8)
    ax.bar(x + width / 2, values_b, width, label=DATE_B_LABEL, alpha=0.8)
    ax.set_xlabel("Balance Range (HBAR)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    return ax


def plot_range_distribution(
    range_df: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    as_percentage: bool = False,
    totals: dict[str, float] | None = None,
) -> plt.Axes:
    """Plot balance-range distribution as counts or percentages."""
    values_a = range_df[f"accounts_{DATE_A}"]
    values_b = range_df[f"accounts_{DATE_B}"]

    ylabel = "Number of Accounts"
    if as_percentage:
        if totals is None:
            raise ValueError("`totals` is required when as_percentage=True")
        values_a = (values_a / totals[DATE_A]) * 100
        values_b = (values_b / totals[DATE_B]) * 100
        ylabel = "Percentage of Total Accounts (%)"

    return _plot_grouped_bars(
        ax=ax,
        x_labels=range_df["range_label"],
        values_a=values_a,
        values_b=values_b,
        title=f"Balance Range Distribution: {title}",
        ylabel=ylabel,
    )


def plot_staking_participation_rate(
    range_all: pd.DataFrame,
    range_staking: pd.DataFrame,
    ax: plt.Axes,
) -> plt.Axes:
    """Plot staking participation by balance range."""
    participation_a = (
        range_staking[f"accounts_{DATE_A}"] / range_all[f"accounts_{DATE_A}"]
    ) * 100
    participation_b = (
        range_staking[f"accounts_{DATE_B}"] / range_all[f"accounts_{DATE_B}"]
    ) * 100
    return _plot_grouped_bars(
        ax=ax,
        x_labels=range_staking["range_label"],
        values_a=participation_a,
        values_b=participation_b,
        title="Staking Participation Rate by Balance Range",
        ylabel="Staking Participation Rate (%)",
    )


def plot_cdf(cdf_df: pd.DataFrame, title: str, ax: plt.Axes) -> plt.Axes:
    """Plot CDF for percentile balances."""
    offset = 0.1
    balance_a = cdf_df[f"balance_{DATE_A}"].replace(0, offset)
    balance_b = cdf_df[f"balance_{DATE_B}"].replace(0, offset)

    ax.plot(
        balance_a, cdf_df["percentile"], marker="o", label=DATE_A_LABEL, linewidth=2
    )
    ax.plot(
        balance_b, cdf_df["percentile"], marker="s", label=DATE_B_LABEL, linewidth=2
    )
    ax.set_xlabel("Balance (HBAR)", fontsize=12)
    ax.set_ylabel("Percentile (%)", fontsize=12)
    ax.set_title(f"CDF: {title}", fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(
        0.02,
        0.02,
        "Note: Zero balances shown as 0.1 HBAR",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.7,
    )
    return ax


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    all_ranges = df[
        df["metric"].str.contains("all.range.") | (df["metric"] == "all.total_hbar")
    ]
    staking_ranges = df[
        df["metric"].str.contains("staking.range.")
        | (df["metric"] == "staking.total_hbar")
    ]
    all_percentiles = df[
        df["metric"].str.contains("all.p") | (df["metric"] == "all.total_accounts")
    ]
    staking_percentiles = df[
        df["metric"].str.contains("staking.p")
        | (df["metric"] == "staking.total_accounts")
    ]

    cdf_all = prepare_cdf_data(all_percentiles)
    cdf_staking = prepare_cdf_data(staking_percentiles)
    range_all = prepare_range_data(all_ranges, include_zero_to_one=False)
    range_all_with_zero = prepare_range_data(all_ranges, include_zero_to_one=True)
    range_staking = prepare_range_data(staking_ranges, include_zero_to_one=True)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_cdf(cdf_all, "All Accounts", ax1)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_cdf(cdf_staking, "Staking Accounts", ax2)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    plot_range_distribution(
        range_all,
        "All Accounts",
        ax3,
        as_percentage=True,
        totals=TOTAL_ACCOUNTS,
    )
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    plot_range_distribution(
        range_staking,
        "Staking Accounts",
        ax4,
        as_percentage=True,
        totals=TOTAL_ACCOUNTS,
    )
    fig4.tight_layout()

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    plot_staking_participation_rate(range_all_with_zero, range_staking, ax5)
    fig5.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
