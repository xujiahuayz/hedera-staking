"""
Demo script to run Hedera staking simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from staking.accounts import Treasury, StakingRewardsPool, NodeRewardsPool
from staking.managers import RewardsEngine, Nodes, Stakers, StakingSystem


# --- Helper utilities for printing ---
def fmt(x: float) -> str:
    return f"{x:,.4f}"


def top_k_dict(d: Dict[int, float], k: int = 5) -> List[Tuple[int, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


def bottom_k_dict(d: Dict[int, float], k: int = 5) -> List[Tuple[int, float]]:
    return sorted(d.items(), key=lambda kv: kv[1])[:k]


@dataclass
class Snapshot:
    day: int
    fees: float
    rs: float
    rn: float
    treasury_balance: float
    staking_pool_balance: float
    node_pool_balance: float | None
    total_staker_balance: float
    total_node_balance: float


def sum_balances(balance_map: Dict[int, float]) -> float:
    return float(sum(balance_map.values()))


# --- Simulation ---
def main():
    rng = np.random.default_rng(123)

    # Simulation size
    NUM_STAKERS = 500
    NUM_NODES = 20
    DAYS = 20

    # Create system accounts
    treasury = Treasury(initial_balance=0.0)
    staking_pool = StakingRewardsPool(initial_balance=250.0)
    node_pool = NodeRewardsPool(initial_balance=100.0)

    # Create stakers and nodes managers
    stakers = Stakers(
        num_stakers=NUM_STAKERS,
        num_nodes=NUM_NODES,
        pareto_shape=2,  # heavier tail -> more whales
        p_switch=0.03,  # small switching probability (sticky)
        rng=rng,
    )

    nodes = Nodes(
        num_stakers=NUM_STAKERS,
        num_nodes=NUM_NODES,
        num_hbars=NUM_STAKERS * 10,  # *10 to scale for min/max stake caps
        node_id_offset=10000,
        pareto_shape=2,
        rng=rng,
    )

    # Rewards + fees engine
    engine = RewardsEngine(
        treasury=treasury,
        staking_pool=staking_pool,
        node_pool=node_pool,
        share_treasury=0.80,  # fee share to 0.0.98
        share_staking_pool=0.10,  # fee share to 0.0.800
        share_node_pool=0.10,  # fee share to 0.0.801
        # reward schedule parameters (from equations in the paper)
        beta=0.50,  # split between node-vs-staker rewards
        epsilon=0.05,
    )

    # Reward schedule parameters (from equations in the paper)
    reward_params = dict(
        param_reward_a=0.01,
        param_reward_b=0.01,
        param_reward_c=5.0,
        param_reward_m=10.0,
    )

    system = StakingSystem(
        stakers=stakers,
        nodes=nodes,
        engine=engine,
        staking_pool=staking_pool,
        node_pool=node_pool,
        reward_params=reward_params,
    )

    # Track history
    history: List[Snapshot] = []

    print("\n=== START SIMULATION ===")
    print(f"stakers={NUM_STAKERS}, nodes={NUM_NODES}, days={DAYS}\n")

    for day in range(DAYS):
        # Pre-step
        treasury_before = treasury.balance
        staking_pool_before = staking_pool.balance
        node_pool_before = node_pool.balance

        total_staker_before = sum_balances(stakers.balance_stakers_account)
        total_node_before = sum_balances(nodes.balance_nodes_account)

        # One day step
        system.step_day()

        # Post-step stats
        treasury_delta = treasury.balance - treasury_before
        staking_pool_delta = staking_pool.balance - staking_pool_before
        node_pool_delta = node_pool.balance - node_pool_before

        # fees inferred (sum of routed shares)
        fees_inferred = treasury_delta + staking_pool_delta + node_pool_delta
        fees_net_signal = fees_inferred

        total_staker_after = sum_balances(stakers.balance_stakers_account)
        total_node_after = sum_balances(nodes.balance_nodes_account)

        eligible_count = len(nodes.eligible_nodes)

        # Largest staker rewards (today) – available from stakers.reward_distribute_staker
        staker_rewards_today = stakers.reward_distribute_staker
        top_staker_rewards = top_k_dict(staker_rewards_today, k=5)

        # Node rewards (today) – nodes.reward_distribute_node
        node_rewards_today = nodes.reward_distribute_node
        top_node_rewards = top_k_dict(node_rewards_today, k=5)

        print(f"--- Day {day} ---")
        print(f"Eligible nodes: {eligible_count}/{NUM_NODES}")
        print(f"Treasury balance:      {fmt(treasury.balance)}")
        print(f"Staking pool balance:  {fmt(staking_pool.balance)}")
        print(f"Node pool balance:     {fmt(node_pool.balance)}")
        print(
            f"Total staker balances: {fmt(total_staker_after)} (Δ {fmt(total_staker_after-total_staker_before)})"
        )
        print(
            f"Total node balances:   {fmt(total_node_after)} (Δ {fmt(total_node_after-total_node_before)})"
        )
        print(f"Net inflow-ish signal (fees - payouts): {fmt(fees_net_signal)}")
        print("Top staker rewards today (staker_id -> reward):")
        for sid, r in top_staker_rewards:
            if r > 0:
                print(f"  {sid:>4} -> {fmt(r)}")
        print("Top node rewards today (node_id -> reward):")
        for nid, r in top_node_rewards:
            if r > 0:
                print(f"  {nid} -> {fmt(r)}")
        print()

        # Save snapshot
        history.append(
            Snapshot(
                day=day,
                fees=fees_net_signal,
                rs=float(sum(staker_rewards_today.values())),
                rn=float(sum(node_rewards_today.values())),
                treasury_balance=float(treasury.balance),
                staking_pool_balance=float(staking_pool.balance),
                node_pool_balance=float(node_pool.balance),
                total_staker_balance=float(total_staker_after),
                total_node_balance=float(total_node_after),
            )
        )

    print("=== END SIMULATION ===\n")

    # Final ranking summaries
    staker_balances = stakers.balance_stakers_account
    node_balances = nodes.balance_nodes_account

    print("Top 10 stakers by balance:")
    for sid, bal in top_k_dict(staker_balances, k=10):
        print(f"  {sid:>4} -> {fmt(bal)}")

    print("\nTop 5 nodes by balance:")
    for nid, bal in top_k_dict(node_balances, k=5):
        print(f"  {nid} -> {fmt(bal)}")

    print("\nBottom 5 nodes by balance:")
    for nid, bal in bottom_k_dict(node_balances, k=5):
        print(f"  {nid} -> {fmt(bal)}")

    # Simple sanity checks
    total_rewards_paid = sum(s.rs + s.rn for s in history)
    print(
        f"\nTotal rewards paid over {DAYS} days (stakers+nodes): {fmt(total_rewards_paid)}"
    )
    print(
        f"Final treasury: {fmt(treasury.balance)} | final 0.0.800: {fmt(staking_pool.balance)} | final 0.0.801: {fmt(node_pool.balance)}"
    )


if __name__ == "__main__":
    main()
