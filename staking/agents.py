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

    def generate_daily_performance(self, day=int):
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
    num_stakers: int
    num_nodes: int

    def __post_init__(self):
        # Initialise balances
        self.agents_node = list(range(10000, 10000 + self.num_nodes))
        self.balance_nodes_intial = np.random.pareto(
            1, self.num_nodes
        )  # intial balance follows the power-law distribution
        self.balance_nodes_account = dict(
            zip(self.agents_node, self.balance_nodes_intial)
        )

    def update_status(self, s_n_network):
        """participant level and total staked from staker"""
        self.parti_level = [
            np.random.normal(0.9, 0.05, self.num_nodes)
        ]  # first assume the particitation level is normal distribution(mean=0.9, std=0.05)

        self.node_stakers_network = s_n_network
        top_nodes = {
            n
            for n, d in self.node_stakers_network.nodes(data=True)
            if d["bipartite"] == 0
        }
        bottom_nodes = set(self.node_stakers_network) - top_nodes
        degS, degN = bipartite.degrees(
            self.node_stakers_network, bottom_nodes, weight="weight"
        )
        self.edge_staking = dict(degN)

    def distribute_rewards(self, rn):
        # balances get updated based on the rewards
        self.reward_distribute_node = {}
        Total_edge_staked = sum(self.edge_staking.values())
        for node in self.agents_node:
            ## if self.edge_staking[node]> Cap_max | < Low_min
            ## if self.parti_level[node]> parti_threshold
            self.reward_distribute_node[node] = rn * (
                self.edge_staking[node] / Total_edge_staked
            )
            self.balance_nodes_account[node] += self.reward_distribute_node[node]


class Stakers:
    def __init__(self, num_stakers, num_nodes):  # ?staker choose one node each time
        # Initialise balances
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

    def network_stakes(self, num_staking_nodes: int) -> nx.Graph:
        # stakers make a decision on whom they want to stake

        self.staking_network = nx.Graph()
        self.staking_network.add_nodes_from(self.agents_staker, bipartite=0)
        self.staking_network.add_nodes_from(self.agents_node, bipartite=1)

        for staker in self.agents_staker:
            staker_to_node = np.random.choice(
                self.agents_node, int(num_staking_nodes), replace=False
            )  # ?link: random OR preferential

            for node in staker_to_node:
                self.staking_network.add_weighted_edges_from(
                    [(staker, node, self.balance_stakers_account[staker])]
                )  # stakers stake out all their balance
        return self.staking_network

    ##def update_stakes(self): #update the selecting and amount of stakeing

    def distribute_rewards(self, rs):
        self.reward_distribute_staker = {}
        # balances get updated based on the rewards
        Sum_stake = sum(self.balance_stakers_account.values())
        for staker in self.agents_staker:
            self.reward_distribute_staker[staker] = rs * (
                self.balance_stakers_account[staker] / Sum_stake
            )
            self.balance_stakers_account[staker] += self.reward_distribute_staker[
                staker
            ]


class Tx_fee:
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
    def __init__(self, pra, prb, prc, prm, ta0, ra0, alpha, beta, epsilon, parameter_l):
        # This function initialises the system of rewards and treasury
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

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
        self.reward_to_stakers = (1 - self.beta) * (reward_t - self.epsilon)

        self.reward_to_nodes = self.beta * (reward_t - self.epsilon)

    def reward_schedule(self, t):
        # From Equation 7
        return (
            self.param_reward_a / t ** (1 / self.param_reward_m)
            + self.param_reward_b * t ** (1 / self.param_reward_m)
            + self.param_reward_c
        )

    def topup(self):
        return 100
