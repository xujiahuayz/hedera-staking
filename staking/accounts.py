"""
This file contains the account types for the Hedera Staking Modelling.
It contains the following class:
- Account: A base class for all accounts
And the following subclasses:
- Staker: user accounts that earn staking rewards (can stake and receive rewards)
- Node: validating node accounts that receive stakes
- StakingRewardsPool: system pool analogous to Hedera 0.0.800
- NodeRewardsPool: system pool analogous to Hedera 0.0.801
- Treasury: system pool analogous to Hedera 0.0.98 (can receive fees and transfer to Reward account)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import TYPE_CHECKING

from staking.constants import RUNTIME_THRESHOLD

if TYPE_CHECKING:
    from staking.managers import Nodes, Stakers


class Account:
    """Base class for all accounts"""

    def __init__(self, account_id: int | str, initial_balance: float = 0.0):
        self.id = account_id
        self.balance = float(initial_balance)
        self.history: list[float] = []

    def update_balance(self, amount: float):
        amount = float(amount)
        self.balance += amount
        self.history.append(amount)


class Staker(Account):
    """User account that can stake and receive staking rewards."""

    def __init__(
        self,
        account_id: int,
        initial_balance: float = 0.0,
        p_switch: float = 0.05,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(account_id, initial_balance)
        if not (0.0 <= p_switch <= 1.0):
            raise ValueError("p_switch must be in [0, 1].")

        self.p_switch = float(p_switch)  # probability of switching a node to stake
        self.rng = rng if rng is not None else np.random.default_rng()

        self.current_target: int | None = None
        self.staked_since_day: int | None = None

    def choose_target(
        self, node_ids: list[int], day: int, chosen_node: int | None = None
    ) -> int:
        day = int(day)
        "Choose one target node to stake. If there's no specific node indicated, a node will be assigned randomly."

        def _set_target(new_target: int):
            if self.current_target != new_target:
                self.current_target = new_target
                self.staked_since_day = day
            elif self.staked_since_day is None:
                self.staked_since_day = day

        if chosen_node is not None:
            if chosen_node not in node_ids:
                raise ValueError("chosen_node must be in node_ids")
            _set_target(int(chosen_node))
            return int(self.current_target)

        if self.current_target is None:
            _set_target(int(self.rng.choice(node_ids)))
            return int(self.current_target)

        if self.rng.random() < self.p_switch:
            _set_target(int(self.rng.choice(node_ids)))

        return int(self.current_target)


class Node(Account):
    """Validating node account with stochastic performance."""

    def __init__(
        self,
        account_id: int | str,
        quality: float = 0.9,
        initial_balance: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        if not (0.0 <= quality <= 1.0):
            raise ValueError("quality must be in [0, 1]")

        super().__init__(account_id, initial_balance)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.quality = float(np.clip(self.rng.normal(loc=quality, scale=0.05), 0, 1))
        self.performance_history: dict[str, dict] = {}

    def generate_daily_performance(self, day: int):
        runtime = float(self.rng.triangular(left=0.0, mode=self.quality, right=1.0))
        self.performance_history[f"day_{day}"] = {
            "day": int(day),
            "runtime": runtime,
            "pass_threshold": runtime >= RUNTIME_THRESHOLD,
        }


class NodeRewardsPool(Account):
    """System pool analogous to Hedera 0.0.801."""

    def __init__(self, account_id: str = "0.0.801", initial_balance: float = 0.0):
        super().__init__(account_id, initial_balance)

    def fund_from_fees(self, amount: float) -> float:
        amount = float(amount)
        if amount <= 0:
            return 0.0
        self.update_balance(amount)
        return amount

    def payout_to_nodes(self, nodes: "Nodes", rn: float) -> float:
        rn = min(float(rn), self.balance)
        if rn <= 0:
            # Ensure downstream reporting doesn't accidentally reuse yesterday's mapping.
            try:
                nodes.distribute_rewards(0.0)
            except Exception:
                pass
            return 0.0

        rewards = nodes.distribute_rewards(rn)
        paid = float(sum(rewards.values())) if rewards is not None else 0.0

        # Defensive clamp against float drift.
        paid = min(paid, rn, self.balance)
        if paid <= 0:
            return 0.0

        self.update_balance(-paid)
        return paid


class StakingRewardsPool(Account):
    """System pool analogous to Hedera 0.0.800."""

    def __init__(self, account_id: str = "0.0.800", initial_balance: float = 250.0):
        super().__init__(account_id, initial_balance)

    def fund_from_fees(self, amount: float) -> float:
        amount = float(amount)
        if amount <= 0:
            return 0.0
        self.update_balance(amount)
        return amount

    def payout_to_stakers(
        self,
        stakers: "Stakers",
        rs: float,
        *,
        day: int,
        staking_network: nx.Graph | None = None,
        eligible_nodes: set[int] | None = None,
        rewardable_stake: dict[int, int] | None = None,
        min_stake_age_days: int = 1,
    ) -> float:
        rs = float(rs)
        if rs <= 0:
            return 0.0

        pay_amount = min(rs, self.balance)
        if pay_amount <= 0:
            return 0.0

        rewards = stakers.distribute_rewards(
            pay_amount,
            day=int(day),
            staking_network=staking_network,
            eligible_nodes=eligible_nodes,
            rewardable_stake=rewardable_stake,
            min_stake_age_days=int(min_stake_age_days),
        )
        paid = float(sum(rewards.values())) if rewards is not None else 0.0

        # Defensive clamp against float drift.
        paid = min(paid, pay_amount, self.balance)
        if paid <= 0:
            return 0.0

        self.update_balance(-paid)
        return paid


class Treasury(Account):
    """Treasury pool analogous to Hedera 0.0.98."""

    def __init__(self, account_id: int | str = "0.0.98", initial_balance: float = 0.0):
        super().__init__(account_id, initial_balance)

    def receive_fees(self, amount: float) -> float:
        amount = float(amount)
        if amount <= 0:
            return 0.0
        self.update_balance(amount)
        return amount
