"""
Managers for the staking system.
- RewardsEngine: fee + rewards logic (NOT an account)
- Nodes: collection manager for nodes (node accounts + eligibility + node rewards)
- Stakers: collection manager for stakers (staker accounts + snapshot graph + staking reward payouts)
- StakingSystem: system that manages the staking process
"""

from __future__ import annotations
import math
import numpy as np
import networkx as nx
from staking.accounts import Treasury, StakingRewardsPool, NodeRewardsPool, Staker, Node


def calculate_transaction_fees(
    t: int, prp: float = 0.3, prq: float = 0.001, Uo: float = 10, fee_tx: float = 1
) -> float:
    """Transaction fee estimation, from the paper."""
    # t = int(t)
    # num_tx = (
    #     Uo
    #     * (1 - math.exp(-(prp + prq) * t))
    #     / (1 + (prp / prq) * (math.exp(-(prp + prq) * t)))
    # )
    # return float(num_tx * fee_tx)
    return 1e-3


class RewardsEngine:
    """Fee + rewards logic."""

    def __init__(
        self,
        treasury: Treasury,
        staking_pool: StakingRewardsPool,
        node_pool: NodeRewardsPool | None = None,
        share_treasury: float = 0.8,
        share_staking_pool: float = 0.1,
        share_node_pool: float = 0.1,
        beta: float = 0.5,
        epsilon: float = 0.05,
    ):
        self.treasury = treasury
        self.staking_pool = staking_pool
        self.node_pool = node_pool

        total = share_treasury + share_staking_pool + share_node_pool
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Fee shares must sum to 1.0")
        self.share_treasury = float(share_treasury)
        self.share_staking_pool = float(share_staking_pool)
        self.share_node_pool = float(share_node_pool)

        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.time = 0

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

    def step(self, **kwargs) -> tuple[float, float]:
        fees = calculate_transaction_fees(t=self.time)

        self.treasury.receive_fees(self.share_treasury * fees)
        self.staking_pool.fund_from_fees(self.share_staking_pool * fees)
        if self.node_pool is not None:
            self.node_pool.fund_from_fees(self.share_node_pool * fees)

        self.time += 1

        kwargs["t"] = self.time
        reward_t = self.reward_schedule(**kwargs)
        distributable = max(0.0, reward_t - self.epsilon)
        rs_target = (1 - self.beta) * distributable
        rn_target = self.beta * distributable
        rs = min(rs_target, self.staking_pool.balance)
        rn = (
            rn_target
            if self.node_pool is None
            else min(rn_target, self.node_pool.balance)
        )
        return float(rs), float(rn)


class Nodes:
    """Collection manager for nodes (node accounts + eligibility + node rewards)."""

    def __init__(
        self,
        num_stakers: int,
        num_nodes: int,
        num_hbars: int | None = None,
        node_id_offset: int = 10000,
        pareto_shape: float = 3.0,
        rng: np.random.Generator | None = None,
    ):
        self.num_stakers = int(num_stakers)
        self.num_nodes = int(num_nodes)
        self.node_id_offset = int(node_id_offset)
        self.num_hbars = (
            int(num_hbars) if num_hbars is not None else self.num_stakers * 10
        )

        self.rng = rng if rng is not None else np.random.default_rng()

        self.agents_node = list(
            range(self.node_id_offset, self.node_id_offset + self.num_nodes)
        )
        self.pareto_shape = float(pareto_shape)
        balance_nodes_initial = self.rng.pareto(self.pareto_shape, self.num_nodes)

        self.nodes: dict[int, Node] = {}
        for node_id, balance in zip(self.agents_node, balance_nodes_initial):
            node_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
            self.nodes[node_id] = Node(
                account_id=node_id,
                initial_balance=float(balance),
                rng=node_rng,
            )
        self.balance_nodes_account = {
            nid: node.balance for nid, node in self.nodes.items()
        }

        self.edge_staking: dict[int, float] = {}
        self.rewardable_stake: dict[int, int] = {}
        self.eligible_nodes: set[int] = set()
        self.reward_distribute_node: dict[int, float] = {}

    def update_status(self, s_n_network: nx.Graph):
        self.edge_staking = {
            node_id: float(s_n_network.degree(node_id, weight="weight"))
            for node_id in self.agents_node
        }

    def compute_eligibility(self, day: int) -> set[int]:
        """Check whether the node is within the eligibility threshold for rewards."""
        max_stake = self.num_hbars / self.num_nodes
        min_stake = max_stake * 0.25

        self.rewardable_stake = {}
        self.eligible_nodes = set()

        for node_id in self.agents_node:
            node = self.nodes[node_id]
            node.generate_daily_performance(day=int(day))
            passed = bool(node.performance_history[f"day_{day}"]["pass_threshold"])

            stake = float(self.edge_staking.get(node_id, 0.0))

            if stake < min_stake:
                stake = 0.0
            else:
                stake = min(stake, max_stake)

            if not passed:
                stake = 0.0

            s_int = int(math.floor(stake))
            self.rewardable_stake[node_id] = s_int
            if s_int > 0:
                self.eligible_nodes.add(node_id)

        return self.eligible_nodes

    def distribute_rewards(self, rn: float) -> dict[int, float]:
        """Distribute the rewards to the nodes."""
        rn = float(rn)
        total_rewardable = sum(self.rewardable_stake.values())
        self.reward_distribute_node = {node_id: 0.0 for node_id in self.agents_node}

        if total_rewardable == 0 or rn <= 0:
            return self.reward_distribute_node

        for node_id in self.agents_node:
            w = self.rewardable_stake.get(node_id, 0)
            reward = rn * (w / total_rewardable) if w > 0 else 0.0
            self.reward_distribute_node[node_id] = reward
            if reward:
                self.nodes[node_id].update_balance(reward)
                self.balance_nodes_account[node_id] = self.nodes[node_id].balance

        return self.reward_distribute_node


class Stakers:
    """Collection manager for stakers (staker accounts + snapshot graph + staking reward payouts)."""

    def __init__(
        self,
        num_stakers: int,
        num_nodes: int,
        node_id_offset: int = 10000,
        pareto_shape: float = 3.0,
        p_switch: float = 0.05,
        rng: np.random.Generator | None = None,
    ):
        self.num_stakers = int(num_stakers)
        self.num_nodes = int(num_nodes)
        self.node_id_offset = int(node_id_offset)
        self.pareto_shape = float(pareto_shape)
        self.p_switch = float(p_switch)

        self.rng = rng if rng is not None else np.random.default_rng()

        self.agents_staker = list(range(self.num_stakers))
        self.agents_node = list(
            range(self.node_id_offset, self.node_id_offset + self.num_nodes)
        )

        balances = self.rng.pareto(self.pareto_shape, self.num_stakers)
        # Scale so mean balance ≈ 10 (since num_hbars = num_stakers * 10)
        balances = 10.0 * balances

        self.stakers: dict[int, Staker] = {}
        for sid, bal in zip(self.agents_staker, balances):
            staker_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
            self.stakers[sid] = Staker(
                account_id=sid,
                initial_balance=float(bal),
                p_switch=self.p_switch,
                rng=staker_rng,
            )

        self.balance_stakers_account = {
            sid: agent.balance for sid, agent in self.stakers.items()
        }
        self.staking_network: nx.Graph = nx.Graph()
        self.reward_distribute_staker: dict[int, float] = {}

    def snapshot_stakes(self, day: int, chosen_node: int | None = None) -> nx.Graph:
        """Snapshot the staking network at a given day."""
        day = int(day)
        if chosen_node is not None and chosen_node not in self.agents_node:
            raise ValueError("chosen_node must be a valid node ID in self.agents_node")

        G = nx.Graph()
        G.add_nodes_from(self.agents_staker, bipartite=0)
        G.add_nodes_from(self.agents_node, bipartite=1)

        for sid in self.agents_staker:
            agent = self.stakers[sid]
            stake_amount = int(math.floor(agent.balance))
            if stake_amount <= 0:
                continue

            nid = agent.choose_target(
                self.agents_node, day=day, chosen_node=chosen_node
            )
            G.add_edge(sid, nid, weight=stake_amount)

        self.staking_network = G
        return G

    def distribute_rewards(
        self,
        rs: float,
        day: int,
        staking_network: nx.Graph | None = None,
        *,
        eligible_nodes: set[int] | None = None,
        rewardable_stake: dict[int, int] | None = None,
        min_stake_age_days: int = 1,
    ) -> dict[int, float]:
        """Distribute the rewards to the stakers."""
        day = int(day)
        min_stake_age_days = int(min_stake_age_days)

        if staking_network is None:
            staking_network = self.staking_network

        rs = float(rs)
        if rs <= 0:
            self.reward_distribute_staker = {sid: 0.0 for sid in self.agents_staker}
            return self.reward_distribute_staker

        # Total raw stake per node (from snapshot)
        node_total_stake: dict[int, int] = {}
        for sid in self.agents_staker:
            nbrs = list(staking_network.neighbors(sid))
            if len(nbrs) != 1:
                continue
            nid = nbrs[0]
            w = int(staking_network[sid][nid].get("weight", 0))
            if w > 0:
                node_total_stake[nid] = node_total_stake.get(nid, 0) + w

        # Effective stake per staker (age + eligibility + oversubscription scaling)
        staker_effective_stake: dict[int, int] = {}

        for sid in self.agents_staker:
            agent = self.stakers[sid]

            if (
                agent.staked_since_day is None
                or (day - agent.staked_since_day) < min_stake_age_days
            ):
                continue

            nbrs = list(staking_network.neighbors(sid))
            if len(nbrs) != 1:
                continue
            nid = nbrs[0]

            if eligible_nodes is not None and nid not in eligible_nodes:
                continue

            w = int(staking_network[sid][nid].get("weight", 0))
            if w <= 0:
                continue

            if rewardable_stake is not None:
                cap = int(rewardable_stake.get(nid, 0))
                tot = int(node_total_stake.get(nid, 0))
                if cap <= 0 or tot <= 0:
                    continue
                scale = min(1.0, cap / tot)
                w_eff = int(math.floor(w * scale))
            else:
                w_eff = w

            if w_eff > 0:
                staker_effective_stake[sid] = w_eff

        total_effective = sum(staker_effective_stake.values())
        if total_effective == 0:
            self.reward_distribute_staker = {sid: 0.0 for sid in self.agents_staker}
            return self.reward_distribute_staker

        rewards: dict[int, float] = {}
        for sid in self.agents_staker:
            w_eff = staker_effective_stake.get(sid, 0)
            reward = rs * (w_eff / total_effective) if w_eff > 0 else 0.0
            rewards[sid] = reward

            if reward:
                self.stakers[sid].update_balance(reward)
            self.balance_stakers_account[sid] = self.stakers[sid].balance

        self.reward_distribute_staker = rewards
        return rewards


class StakingSystem:
    """System that manages the staking process."""

    def __init__(
        self,
        stakers: Stakers,
        nodes: Nodes,
        engine: RewardsEngine,
        staking_pool: StakingRewardsPool,
        node_pool: NodeRewardsPool | None = None,
        reward_params: dict[str, float] | None = None,
    ):
        self.stakers = stakers
        self.nodes = nodes
        self.engine = engine
        self.staking_pool = staking_pool
        self.node_pool = node_pool
        self.reward_params = dict(reward_params) if reward_params is not None else {}
        self.day = 0

        if stakers.node_id_offset != nodes.node_id_offset:
            raise ValueError("node_id_offset mismatch between Stakers and Nodes")
        if stakers.num_nodes != nodes.num_nodes:
            raise ValueError("num_nodes mismatch between Stakers and Nodes")

    def step_day(self):
        """Step forward one day."""
        G = self.stakers.snapshot_stakes(day=self.day)
        self.nodes.update_status(G)
        eligible = self.nodes.compute_eligibility(day=self.day)

        rs, rn = self.engine.step(**self.reward_params)

        self.staking_pool.payout_to_stakers(
            self.stakers,
            rs,
            day=self.day,
            staking_network=G,
            eligible_nodes=eligible,
            rewardable_stake=self.nodes.rewardable_stake,  # <-- correct instance dict
        )

        if self.node_pool is not None:
            self.node_pool.payout_to_nodes(self.nodes, rn)
        else:
            self.nodes.distribute_rewards(rn)

        self.day += 1
