import matplotlib.pyplot as plt

import networkx as nx
from staking.agents import HederaSystem, Tx_fee, HBar


T = HederaSystem()

Stakers_Balance_t = {}
Nodes_Balance_t = {}
Reward_Account_t = []
Treasury_Account_t = []

for t in range(1, 50):
    T.iterate()
    Stakers_Balance_t[t] = T.stakers.balance_stakers_account.copy()
    Nodes_Balance_t[t] = T.nodes.balance_nodes_account.copy()
    Reward_Account_t.append(T.hbar.reward)
    Treasury_Account_t.append(T.hbar.treasury)


# %%
plt.plot(Reward_Account_t)

# %%
plt.plot(Treasury_Account_t)

# %%
Reward_node = {}
for node in Nodes_Balance_t[1].keys():
    for t in Nodes_Balance_t.keys():
        Reward_node[node] = Reward_node.get(node, []) + [Nodes_Balance_t[t][node]]
    plt.plot(Reward_node[node], label=node)
plt.legend()

# %%
Reward_staker = {}
for staker in Stakers_Balance_t[1].keys():
    for t in Stakers_Balance_t.keys():
        Reward_staker[staker] = Reward_staker.get(staker, []) + [
            Stakers_Balance_t[t][staker]
        ]
    plt.plot(Reward_staker[staker], label=staker)
# plt.legend()

# %%
### Test amount of Tx_fee

fee_t = []
for t in range(50):
    fee_t.append(Tx_fee(t).calculate_sum_fee())

plt.plot(fee_t)

# %%
###Test Totoal_reward at each time t

T_r = HBar(
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
)

reward_t = []
for t in range(1, 50):
    reward_t.append(T_r.reward_schedule(t))

plt.plot(reward_t)

# %%
###Test staking network

sn_network = T.S_N_network
plt.figure(figsize=(5, 6), dpi=200)

color_map = []
for node in sn_network.nodes:
    if node < 10000:
        color_map.append("green")
    else:
        color_map.append("tab:blue")

pos = {node: [0, i] for i, node in enumerate(T.stakers.agents_staker)}
pos.update({node: [1, i] for i, node in enumerate(T.stakers.agents_node)})
# print(pos)
nx.draw(
    sn_network,
    pos=pos,
    node_color=color_map,
    with_labels=True,
    node_size=100,
    font_size=8,
)
