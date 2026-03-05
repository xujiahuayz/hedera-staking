"""Managers for the staking system."""

from __future__ import annotations

import math

import numpy as np

from staking.accounts import (
    FeeCollection,
    Node,
    NodeRewardsPool,
    Staker,
    StakingRewardsPool,
    Treasury,
)


class StakingEnvironment:
    """Unified environment for stakers + nodes state and reward transitions."""

    def __init__(
        self,
        num_stakers: int,
        num_nodes: int,
        num_hbars: int = 42000000000,
        staker_pareto_shape: float = 3.0,
        node_pareto_shape: float = 3.0,
        treasury: Treasury | None = None,
        staking_pool: StakingRewardsPool | None = None,
        node_pool: NodeRewardsPool | None = None,
        fee_collection: FeeCollection | None = None,
        reward_params: dict[str, float] | None = None,
        node_fee_per_tx: float = 1e-5,
        staker_activity_min: float = 0.1,
        staker_activity_scale: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.num_stakers = int(num_stakers)
        self.num_nodes = int(num_nodes)
        self.num_hbars = int(num_hbars)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.day = 0
        self.treasury = (
            treasury if treasury is not None else Treasury(initial_balance=3284822.0)
        )
        self.staking_pool = StakingRewardsPool(env=self, initial_balance=215204097.0)

        self.node_pool = node_pool
        self.fee_collection = (
            fee_collection if fee_collection is not None else FeeCollection(env=self)
        )
        self.reward_params = dict(reward_params)
        self.node_fee_per_tx = float(node_fee_per_tx)
        self.staker_activity_min = float(staker_activity_min)
        self.staker_activity_scale = float(staker_activity_scale)
        self.node_ids = list(range(0, self.num_nodes))

        self.staking_pool.env = self
        self.fee_collection.env = self

        staker_balances = 10.0 * self.rng.pareto(
            float(staker_pareto_shape), self.num_stakers
        )
        node_balances = self.rng.pareto(float(node_pareto_shape), self.num_nodes)

        self.stakers: dict[int, Staker] = {}
        for sid, bal in zip(range(self.num_stakers), staker_balances):
            staker_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
            activity_rate = max(
                self.staker_activity_min,
                self.staker_activity_scale * float(np.log1p(max(float(bal), 0.0))),
            )
            self.stakers[sid] = Staker(
                env=self,
                account_id=sid,
                initial_balance=float(bal),
                activity_rate=activity_rate,
                rng=staker_rng,
            )

        self.nodes: dict[int, Node] = {}
        for nid, bal in zip(self.node_ids, node_balances):
            self.nodes[nid] = Node(
                env=self,
                account_id=nid,
                initial_balance=float(bal),
            )

        self.balance_stakers_account = {
            sid: s.balance for sid, s in self.stakers.items()
        }
        self.balance_nodes_account = {nid: n.balance for nid, n in self.nodes.items()}
        self.stake_map: dict[int, tuple[int, int]] = {}
        self.staking_amount_map: dict[int, float] = {}
        self.rewardable_stake: dict[int, int] = {}
        self.node_payments_info: dict[int, float] = {nid: 0.0 for nid in self.node_ids}
        self.reward_distribute_staker: dict[int, float] = {}
        self.reward_distribute_node: dict[int, float] = {}

    def __repr__(self) -> str:
        return (
            f"StakingEnvironment(day={self.day}, "
            f"treasury_balance={self.treasury.balance:.4f}, "
            f"staking_pool_balance={self.staking_pool.balance:.4f}, "
            f"node_pool_balance={self.node_pool.balance:.4f})"
        )

    @staticmethod
    def calculate_network_service_fees(
        t: int, prp: float = 0.3, prq: float = 0.001, Uo: float = 10, fee_tx: float = 1
    ) -> float:
        """Estimate network + service fees that flow into 0.0.802."""
        t = int(t)
        num_tx = (
            Uo
            * (1 - math.exp(-(prp + prq) * t))
            / (1 + (prp / prq) * (math.exp(-(prp + prq) * t)))
        )
        return float(num_tx * fee_tx)

    def calculate_node_fees(self, t: int) -> dict[int, float]:
        """Estimate per-node fees accumulated in NodePaymentsInfo.

        Node fees are proxied from heterogeneous per-staker transaction volume:
        each staker samples a daily transaction count, and those fees are
        attributed to the node they are currently staking to.
        """
        node_fees = {nid: 0.0 for nid in self.node_ids}
        for sid in range(self.num_stakers):
            stake_info = self.stake_map.get(sid)
            if stake_info is None:
                continue
            nid = int(stake_info[0])
            tx_count = self.stakers[sid].simulate_daily_transactions(day=int(t))
            node_fees[nid] += float(tx_count) * self.node_fee_per_tx

        return node_fees

    def build_stake_map(
        self, day: int | None = None, chosen_node: int | None = None
    ) -> dict[int, tuple[int, int]]:
        """Build staker -> (node, stake) mapping as sid -> (nid, stake_amount)."""
        day = self.day if day is None else int(day)
        if chosen_node is not None and chosen_node not in self.node_ids:
            raise ValueError("chosen_node must be a valid node ID in self.agents_node")

        stake_map: dict[int, tuple[int, int]] = {}

        for sid in range(self.num_stakers):
            staker = self.stakers[sid]
            stake_amount = int(math.floor(staker.balance))
            if stake_amount <= 0:
                continue
            nid = staker.choose_node(self.node_ids, chosen_node=chosen_node)
            stake_map[sid] = (int(nid), int(stake_amount))

        self.stake_map = stake_map
        return stake_map

    def update_node_status(
        self,
        stake_map: dict[int, tuple[int, int]] | None = None,
        day: int | None = None,
    ):
        day = self.day if day is None else int(day)
        stake_map = self.stake_map if stake_map is None else stake_map

        staking_amount_map = {nid: 0.0 for nid in self.node_ids}
        for sid in range(self.num_stakers):
            stake_info = stake_map.get(sid)
            if stake_info is None:
                continue
            nid, stake_amount = stake_info
            if stake_amount > 0:
                staking_amount_map[int(nid)] += float(stake_amount)
        self.staking_amount_map = staking_amount_map
        self.rewardable_stake = {}
        for nid in self.node_ids:
            node = self.nodes[nid]
            staked_amount = float(self.staking_amount_map.get(nid, 0.0))
            self.rewardable_stake[nid] = node.compute_stake(
                num_hbars=self.num_hbars,
                num_nodes=self.num_nodes,
                staked_amount=staked_amount,
            )

    def distribute_node_rewards(self, rn: float) -> dict[int, float]:
        """
        Distribute node rewards based on the node's rewardable stake and the node's payment info.
        """
        rn = float(rn)
        total_rewardable = sum(self.rewardable_stake.values())
        self.reward_distribute_node = {
            nid: float(self.node_payments_info.get(nid, 0.0)) for nid in self.node_ids
        }

        for nid, amount in self.reward_distribute_node.items():
            if amount:
                self.nodes[nid].add_to_balance(amount)
                self.balance_nodes_account[nid] = self.nodes[nid].balance

        if rn <= 0 or total_rewardable == 0:
            self.node_payments_info = {nid: 0.0 for nid in self.node_ids}
            return self.reward_distribute_node

        for nid in self.node_ids:
            w = self.rewardable_stake.get(nid, 0)
            reward = rn * (w / total_rewardable)
            self.reward_distribute_node[nid] += reward
            if reward:
                self.nodes[nid].add_to_balance(reward)
                self.balance_nodes_account[nid] = self.nodes[nid].balance
        self.node_payments_info = {nid: 0.0 for nid in self.node_ids}
        return self.reward_distribute_node

    def distribute_staker_rewards(
        self,
        rs: float,
        day: int | None = None,
        stake_map: dict[int, tuple[int, int]] | None = None,
        *,
        rewardable_stake: dict[int, int] | None = None,
    ) -> dict[int, float]:
        day = self.day if day is None else int(day)
        stake_map = self.stake_map if stake_map is None else stake_map

        rs = float(rs)

        node_total_stake: dict[int, int] = {}
        for sid in range(self.num_stakers):
            stake_info = stake_map.get(sid)
            if stake_info is None:
                continue
            nid, w = stake_info
            if w > 0:
                node_total_stake[nid] = node_total_stake.get(nid, 0) + w

        effective: dict[int, int] = {}
        for sid in range(self.num_stakers):
            stake_info = stake_map.get(sid)
            if stake_info is None:
                continue
            nid, w = stake_info
            if w <= 0:
                continue

            if rewardable_stake is not None:
                cap = int(rewardable_stake.get(nid, 0))
                tot = int(node_total_stake.get(nid, 0))
                if cap <= 0 or tot <= 0:
                    continue
                w = int(math.floor(w * min(1.0, cap / tot)))
            if w > 0:
                effective[sid] = w

        total_effective = sum(effective.values())
        if total_effective == 0:
            self.reward_distribute_staker = {
                sid: 0.0 for sid in range(self.num_stakers)
            }
            return self.reward_distribute_staker

        rewards: dict[int, float] = {}
        for sid in range(self.num_stakers):
            w = effective.get(sid, 0)
            reward = rs * (w / total_effective) if w > 0 else 0.0
            rewards[sid] = reward
            if reward:
                self.stakers[sid].add_to_balance(reward)
            self.balance_stakers_account[sid] = self.stakers[sid].balance

        self.reward_distribute_staker = rewards
        return rewards

    @property
    def state_summary(self) -> dict[str, int | float | None]:
        """Return a summary of the current environment state."""
        return {
            "day": self.day,
            "treasury_balance": float(self.treasury.balance),
            "staking_pool_balance": float(self.staking_pool.balance),
            "node_pool_balance": (
                float(self.node_pool.balance) if self.node_pool is not None else None
            ),
            "total_staker_balance": float(sum(self.balance_stakers_account.values())),
            "total_node_balance": float(sum(self.balance_nodes_account.values())),
        }

    def step_day(self) -> tuple[float, float]:
        stake_map = self.build_stake_map(day=self.day)
        self.update_node_status(stake_map=stake_map, day=self.day)

        network_service_fees = self.calculate_network_service_fees(t=self.day)
        node_fees = self.calculate_node_fees(t=self.day)
        self.node_payments_info = {
            nid: float(node_fees.get(nid, 0.0)) for nid in self.node_ids
        }

        self.fee_collection.collect_fees(network_service_fees)
        _, rs, rn = self.fee_collection.route_fees(
            self.treasury,
            self.staking_pool,
            self.node_pool,
        )
        scheduled_rs = self.staking_pool.reward_schedule(
            t=self.day + 1, **self.reward_params
        )
        rs = min(float(scheduled_rs), float(self.staking_pool.balance))

        self.staking_pool.payout_to_stakers(
            rs=rs,
            day=self.day,
            stake_map=stake_map,
            rewardable_stake=self.rewardable_stake,
        )
        if self.node_pool is not None:
            self.node_pool.payout_to_nodes(rn=rn, env=self)
        else:
            self.distribute_node_rewards(rn)
        self.day += 1
        return rs, rn


def main(
    days: int = 20, num_stakers: int = 120000, num_nodes: int = 32, seed: int = 123
) -> list[dict[str, int | float | None]]:
    rng = np.random.default_rng(seed)
    treasury = Treasury(initial_balance=3284822.0)
    node_pool = NodeRewardsPool(initial_balance=993358.0)
    reward_params = {
        "param_reward_a": 0.01,
        "param_reward_b": 0.01,
        "param_reward_c": 5.0,
        "param_reward_m": 10.0,
    }
    env = StakingEnvironment(
        num_stakers=num_stakers,
        num_nodes=num_nodes,
        num_hbars=42000000000,
        staker_pareto_shape=2.0,
        node_pareto_shape=2.0,
        treasury=treasury,
        staking_pool=StakingRewardsPool(initial_balance=215204097.0),
        node_pool=node_pool,
        fee_collection=FeeCollection(
            share_staking_pool=0.10,
            share_node_pool=0.10,
        ),
        reward_params=reward_params,
        node_fee_per_tx=1e-5,
        staker_activity_min=0.1,
        staker_activity_scale=1.0,
        rng=rng,
    )

    history: list[dict[str, int | float | None]] = []
    print(
        f"\n=== START SIMULATION stakers={num_stakers}, nodes={num_nodes}, days={days} ===\n"
    )
    for day in range(days):
        rs, rn = env.step_day()
        total_staker_after = float(sum(env.balance_stakers_account.values()))
        total_node_after = float(sum(env.balance_nodes_account.values()))
        print(env)
        history.append(
            {
                **env.state_summary,
                "day": day,
                "rs": float(rs),
                "rn": float(rn),
                "total_staker_balance": total_staker_after,
                "total_node_balance": total_node_after,
            }
        )

    print("\n=== END SIMULATION ===")

    return history


if __name__ == "__main__":
    main()
