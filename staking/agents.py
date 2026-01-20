from typing import Any


from dataclasses import dataclass
import numpy as np
import math
import numpy.random as rnd

import networkx as nx
from networkx.algorithms import bipartite

from staking.constants import RUNTIME_THRESHOLD

# from dataclasses import dataclass, field


class ValidatingNode:
    """A validating node, characterized by its quality, that receive stakes"""

    def __init__(self, quality: float = 0.9):
        if not 0 <= quality <= 1:
            raise ValueError("`quality` must be between 0 and 1")
        self.quality = quality
        self.performance_history = {}

    def generate_daily_performance(self, day: int):
        """randomize everyday's runtime based on the node's `quality` at a given `day`"""
        runtime = rnd.triangular(left=0, mode=self.quality, right=1)
        self.performance_history.update(
            {
                f"day_{day}": {
                    "day": day,
                    "runtime": runtime,
                    "pass_threshold": runtime >= RUNTIME_THRESHOLD,
                }
            }
        )


@dataclass
class Nodes:
    """Nodes, characterized by its quality, that receive stakes"""

    num_stakers: int
    num_nodes: int
    num_hbars: int

    def __post_init__(self):
        # Initialise balances
        self.agents_node = list(range(10000, 10000 + self.num_nodes))
        self.balance_nodes_intial = np.random.pareto(
            1, self.num_nodes
        )  # Modelling assumption: intial balance follows the power-law distribution, heavy-tailed distribution, i.e. most nodes have a small balance, but a few nodes have a large balance.
        self.balance_nodes_account = dict(
            zip(self.agents_node, self.balance_nodes_intial)
        )

    def update_status(self, s_n_network):
        """Generates a “participation level” per node, and calculates the total stake each node recieve."""

        # Reads the bipartite staking graph and computes stake per node
        self.node_stakers_network = s_n_network
        top_nodes = {
            n
            for n, d in self.node_stakers_network.nodes(data=True)
            if d["bipartite"] == 0
        }
        bottom_nodes = set(self.node_stakers_network) - top_nodes
        degS, degN = bipartite.degrees(
            self.node_stakers_network, bottom_nodes, weight="weight"
        )  # computes the weighted degree (sum of edge weights)
        self.edge_staking = dict(degN)  # total HBAR staked to that node

    def distribute_rewards(self, rn: float):
        """Split the total reward pot rn across nodes in proportion to their rewardable stake, then add the result to each node’s balance."""
        self.reward_distribute_node = {}
        parti_threshold = 0.8
        max_stake = (
            self.num_hbars / self.num_nodes
        )  # The maximum stake threshold for each node is the total number of HBAR divided by the total number of nodes in the network
        min_stake = (
            max_stake * 0.25
        )  # The minimum node stake threshold value is be 1/4 of the maximum node stake value.

        # Modelling assumption: the particitation level is normal distribution(mean=0.9, std=0.05), creates random node participation rates
        self.parti_level = np.random.normal(0.9, 0.05, self.num_nodes)

        # Rewardable stake per node (apply caps)
        rewardable_stake = {}
        for node in self.agents_node:
            stake = self.edge_staking.get(node, 0)

            if stake < min_stake:
                stake = 0
            else:
                stake = min(stake, max_stake)

            # Participation gating (Modelling assumption: if the node's participation level is less than the participation threshold, the node is not eligible for rewards)
            idx = node - 10000
            if self.parti_level[idx] < parti_threshold:
                stake = 0

            # Hedera uses whole HBAR for staking math
            rewardable_stake[node] = math.floor(stake)

        total_rewardable = sum(rewardable_stake.values())

        # If nobody is eligible, no rewards distributed
        if total_rewardable == 0:
            for node in self.agents_node:
                self.reward_distribute_node[node] = 0
            return

        # Distribute reward pot (rn) proportionally
        for node in self.agents_node:
            self.reward_distribute_node[node] = rn * (
                rewardable_stake[node] / total_rewardable
            )
            self.balance_nodes_account[node] += self.reward_distribute_node[
                node
            ]  # Add the reward that node earned today into its account balance.


class Stakers:
    def __init__(self, num_stakers, num_nodes):  # staker choose one node each time
        """Initialise balances"""
        self.num_stakers = num_stakers

        self.agents_staker = list(range(self.num_stakers))
        self.balance_stakers_intial = np.random.pareto(
            1, self.num_stakers
        )  # intial balance follows the power-law distribution
        self.balance_stakers_account = dict(
            zip(self.agents_staker, self.balance_stakers_intial)
        )

        self.num_nodes = num_nodes
        self.agents_node = list(range(10000, 10000 + self.num_nodes))

    def network_stakes(self, chosen_node: int = None) -> nx.Graph:
        """Stakers can make a decision on whom they want to stake, else they choose a random node"""

        self.staking_network = nx.Graph()
        self.staking_network.add_nodes_from(self.agents_staker, bipartite=0)
        self.staking_network.add_nodes_from(self.agents_node, bipartite=1)

        # Validate chosen_node if provided
        if chosen_node is not None and chosen_node not in self.agents_node:
            raise ValueError("chosen_node must be a valid node ID in self.agents_node")

        for staker in self.agents_staker:

            staker_to_node = (
                np.random.choice(self.agents_node)
                if chosen_node is None
                else chosen_node
            )  # if the staker doesnt choose a node, they choose a random node

            stake_amount = math.floor(
                self.balance_stakers_account[staker]
            )  # whole HBAR

            if stake_amount > 0:
                self.staking_network.add_weighted_edges_from(
                    [(staker, staker_to_node, stake_amount)]
                )  # stakers stake out all their balance (rounded down to whole HBAR)
        return self.staking_network

    def distribute_rewards(self, staking_network: nx.Graph, rs: float):
        """
        Distribute total staker rewards rs based on where stakers actually staked.
        Each staker has exactly ONE edge to ONE node.
        """

        # Compute total stake per node
        node_total_stake = {}
        for staker in self.agents_staker:
            # Each staker should have exactly one neighbor node
            neighbors = list(staking_network.neighbors(staker))
            if not neighbors:
                continue
            node = neighbors[0]
            stake = staking_network[staker][node]["weight"]
            node_total_stake[node] += stake

        total_stake_all_nodes = sum(node_total_stake.values())
        if total_stake_all_nodes == 0:
            return

        # Give each node a share of rs proportional to its total stake
        node_reward = {
            node: rs * (stake / total_stake_all_nodes)
            for node, stake in node_total_stake.items()
        }

        # Within each node, split node_reward among stakers proportional to their stake
        self.reward_distribute_staker = {}
        for staker in self.agents_staker:
            neighbors = list(staking_network.neighbors(staker))
            if not neighbors:
                self.reward_distribute_staker[staker] = 0
                continue

            node = neighbors[0]
            stake = staking_network[staker][node]["weight"]

            if node_total_stake[node] == 0:
                self.reward_distribute_staker[staker] = 0
                continue

            reward = node_reward[node] * (
                stake / node_total_stake[node]
            )  # Split the reward that node earned among stakers proportional to their stake.
            self.reward_distribute_staker[staker] = reward
            self.balance_stakers_account[
                staker
            ] += reward  # Add the reward that staker earned today into its account balance.


class Tx_fee:
    """“Model for transaction volume growth and fee revenue”"""

    def __init__(
        self, t, prp=0.3, prq=0.001, Uo=10, fee_tx=1
    ):  ##? fixed Tx_fee  further discuss the parameter
        self.num_tx = (
            Uo
            * (1 - math.exp(-(prp + prq) * t))
            / (1 + (prp / prq) * (math.exp(-(prp + prq) * t)))
        )

        self.fee_tx = fee_tx

    def calculate_sum_fee(self):

        self.sum_fee = self.num_tx * self.fee_tx
        return self.sum_fee


class HBar:
    """
    Model for Hedera's reward system, which includes the treasury and reward accounts.
    """

    def __init__(self, pra, prb, prc, prm, ta0, ra0, alpha, beta, epsilon, parameter_l):
        # This function initialises the system of rewards and treasury
        self.alpha = alpha  # Controls how transaction fees are split
        self.beta = beta  # Controls how the daily reward payout is split
        self.epsilon = epsilon  # Constant “cost” subtracted from rewards:

        self.treasury = ta0
        self.reward = ra0
        self.param_reward_a = pra
        self.param_reward_b = prb
        self.param_reward_c = prc
        self.param_reward_m = prm
        self.time = 0
        self.parameter_l = parameter_l  ## ? when treasure transfer to reward

    def iterate(self):
        # This function is called each time-step to update the internal variables of the system
        print("time:", self.time)
        self.transaction_fees = Tx_fee(t=self.time).calculate_sum_fee()
        # print(self.transaction_fees)

        # Equation 1
        self.flow_fees_to_treasury = self.alpha * self.transaction_fees

        # Equation 4
        self.flow_fees_to_reward = (1 - self.alpha) * self.transaction_fees

        # Equation 2
        if rnd.random() < self.parameter_l:
            self.flow_treasury_to_reward = rnd.uniform(
                0, self.treasury + self.flow_fees_to_treasury
            )
        else:
            self.flow_treasury_to_reward = 0

        # Equation 3
        self.treasury += self.flow_fees_to_treasury - self.flow_treasury_to_reward

        # Equation 5
        self.reward += self.flow_fees_to_reward + self.flow_treasury_to_reward

        self.time += 1

        # The following computes the reward given in day t
        reward_t = self.reward_schedule(self.time)

        if reward_t <= self.reward:
            self.reward -= reward_t
        else:
            # Money is not sufficient in the reward account
            print("FAIL", self.time)
            # Triggers top-up
            self.reward += self.topup()

        # From equation 11
        self.reward_to_stakers = (1 - self.beta) * (
            reward_t - self.epsilon
        )  # how much to distribute to stakers today

        self.reward_to_nodes = self.beta * (
            reward_t - self.epsilon
        )  # how much to distribute to nodes today

    def reward_schedule(self, t):
        # From Equation 7
        return (
            self.param_reward_a / t ** (1 / self.param_reward_m)
            + self.param_reward_b * t ** (1 / self.param_reward_m)
            + self.param_reward_c
        )

    def topup(self):
        return 100
