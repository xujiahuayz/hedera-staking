"""
This file contains the account types for the Hedera Staking Modelling.
It contains the following class:
- Account: A base class for all accounts
And the following subclasses:
- Staker: user accounts that earn staking rewards (can stake and receive rewards)
- Node: validating node accounts that receive stakes
- StakingRewardsPool: system pool analogous to Hedera 0.0.800
- NodeRewardsPool: system pool analogous to Hedera 0.0.801
- FeeCollection: system fee collection account analogous to Hedera 0.0.802
- Treasury: system pool analogous to Hedera 0.0.98 (can receive fees and transfer to Reward account)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from staking.stakingenv import StakingEnvironment


class Account:
    """Base class for all accounts"""

    def __init__(self, account_id: int | str, initial_balance: float = 0.0):
        self.id = account_id
        self.balance = float(initial_balance)

    def add_to_balance(self, amount: float):
        amount = float(amount)
        self.balance += amount


class Staker(Account):
    """User account that can stake and receive staking rewards."""

    def __init__(
        self,
        account_id: int,
        initial_balance: float = 0.0,
        activity_rate: float | None = None,
        rng: np.random.Generator | None = None,
        env: StakingEnvironment = None,
    ):
        self.env = env
        super().__init__(account_id, initial_balance)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Use a heterogeneous baseline activity rate so higher-balance stakers
        # tend to generate more transactions without making activity deterministic.
        if activity_rate is None:
            activity_rate = max(0.1, float(np.log1p(max(initial_balance, 0.0))))
        self.activity_rate = float(activity_rate)
        self.daily_transactions_history: dict[int, int] = {}

        self.current_target: int | None = None
        self.staked_since_day: int | None = None

    def choose_node(self, node_ids: list[int], chosen_node: int | None = None) -> int:
        "Choose one target node to stake. If there's no specific node indicated, a node will be assigned randomly."

        def _set_target(new_target: int):
            if self.current_target != new_target:
                self.current_target = new_target

        if chosen_node is not None:
            if chosen_node not in node_ids:
                raise ValueError("chosen_node must be in node_ids")
            _set_target(int(chosen_node))
            return int(self.current_target)

        if self.current_target is None:
            _set_target(int(self.rng.choice(node_ids)))
            return int(self.current_target)

        return int(self.current_target)

    def simulate_daily_transactions(self, day: int) -> int:
        """Sample this staker's daily transaction count."""
        tx_count = int(self.rng.poisson(self.activity_rate))
        self.daily_transactions_history[int(day)] = tx_count
        return tx_count


class Node(Account):
    """Calculate the final stake for a node."""

    def __init__(
        self,
        account_id: int | str,
        initial_balance: float = 0.0,
        env: StakingEnvironment = None,
    ):
        self.env = env
        super().__init__(account_id, initial_balance)

    def compute_stake(
        self, num_hbars: int, num_nodes: int, staked_amount: float
    ) -> int:
        """Compute this node's rewardable stake using the network caps.
        staked_amount is the total amount of stakers pointed to this node."""
        max_stake = float(num_hbars) / float(num_nodes)
        stake = min(float(staked_amount), max_stake)
        return int(np.floor(max(0.0, stake)))


class FeeCollection(Account):
    """System fee collection account analogous to Hedera 0.0.802. It collects fees and routes them to the treasury, staking pool, and node pool."""

    def __init__(
        self,
        account_id: str = "0.0.802",
        initial_balance: float = 0.0,
        share_staking_pool: float = 0.1,
        share_node_pool: float = 0.1,
        env: StakingEnvironment = None,
    ):
        self.env = env
        super().__init__(account_id, initial_balance)

        self.share_staking_pool = float(share_staking_pool)
        self.share_node_pool = float(share_node_pool)
        self.share_treasury = 1 - self.share_staking_pool - self.share_node_pool

    def collect_fees(self, amount: float) -> float:
        amount = float(amount)
        if amount <= 0:
            return 0.0
        self.add_to_balance(amount)
        return amount

    def route_fees(
        self,
        treasury: Treasury,
        staking_pool: StakingRewardsPool,
        node_pool: NodeRewardsPool,
    ) -> tuple[float, float, float]:
        if self.balance <= 0:
            return 0.0, 0.0, 0.0

        fees = float(self.balance)
        t_amt = self.share_treasury * fees
        s_amt = self.share_staking_pool * fees
        n_amt = self.share_node_pool * fees

        treasury.receive_fees(t_amt)
        staking_pool.fund_from_fees(s_amt)
        if node_pool is not None:
            node_pool.fund_from_fees(n_amt)

        self.add_to_balance(-fees)
        return float(t_amt), float(s_amt), float(n_amt)


class NodeRewardsPool(Account):
    """System pool analogous to Hedera 0.0.801."""

    def __init__(
        self,
        account_id: str = "0.0.801",
        initial_balance: float = 0.0,
        env: StakingEnvironment = None,
    ):
        self.env = env
        super().__init__(account_id, initial_balance)

    def fund_from_fees(self, amount: float) -> float:
        amount = float(amount)
        if amount <= 0:
            return 0.0
        self.add_to_balance(amount)
        return amount

    def payout_to_nodes(self, env: StakingEnvironment, rn: float) -> float:
        rn = min(float(rn), self.balance)

        rewards = env.distribute_node_rewards(rn)
        paid = float(sum(rewards.values()))

        paid = min(paid, rn, self.balance)
        if paid <= 0:
            return 0.0

        self.add_to_balance(-paid)
        return paid


class StakingRewardsPool(Account):
    """System pool analogous to Hedera 0.0.800."""

    def __init__(
        self,
        env: StakingEnvironment = None,
        account_id: str = "0.0.800",
        initial_balance: float = 250.0,
    ):
        super().__init__(account_id, initial_balance)
        self.env = env

    def fund_from_fees(self, amount: float) -> float:
        amount = float(amount)
        if amount <= 0:
            return 0.0
        self.add_to_balance(amount)
        return amount

    @staticmethod
    def reward_schedule(
        *,
        param_reward_a: float = 0.01,
        param_reward_b: float = 0.01,
        param_reward_c: float = 5.0,
        param_reward_m: float = 10.0,
        t: int,
    ) -> float:
        t = max(1, int(t))
        return (
            param_reward_a / (t ** (1 / param_reward_m))
            + param_reward_b * (t ** (1 / param_reward_m))
            + param_reward_c
        )

    def payout_to_stakers(
        self,
        rs: float,
        *,
        day: int,
        staker_node_stake_map: dict[int, tuple[int, int]],
        rewardable_stake: dict[int, int] | None = None,
    ) -> float:
        rs = float(rs)
        pay_amount = min(rs, self.balance)

        rewards = self.env.distribute_staker_rewards(
            pay_amount,
            day=int(day),
            staker_node_stake_map=staker_node_stake_map,
            rewardable_stake=rewardable_stake,
        )
        paid = float(sum(rewards.values()))

        paid = min(paid, pay_amount, self.balance)

        self.add_to_balance(-paid)
        return paid


class Treasury(Account):
    """Treasury pool analogous to Hedera 0.0.98."""

    def __init__(
        self,
        account_id: int | str = "0.0.98",
        initial_balance: float = 0.0,
        env: StakingEnvironment = None,
    ):
        self.env = env
        super().__init__(account_id, initial_balance)

    def receive_fees(self, amount: float) -> float:
        amount = float(amount)
        self.add_to_balance(amount)
        return amount
