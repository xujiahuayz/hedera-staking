from staking.agents import HBar, Stakers, Nodes


class HederaSystem:
    def __init__(self):

        self.hbar = HBar(
            pra=0,
            prb=0,
            prc=5,
            prm=10,
            ta0=100,
            ra0=10,
            alpha=0.1,
            beta=0.5,
            epsilon=0.05,
            parameter_l=0.1,
        )  # add all parameters.

        self.stakers = Stakers(40, 10)  # Add initialisation num_staker=40, num_node=10

        self.nodes = Nodes(40, 10)

    def iterate(self):

        self.S_N_network = self.stakers.network_stakes(
            num_staking_nodes=1
        )  # each staker only select one node

        self.nodes.update_status(self.S_N_network)

        self.hbar.iterate()

        self.stakers.distribute_rewards(self.hbar.reward_to_stakers)

        self.nodes.distribute_rewards(self.hbar.reward_to_nodes)
