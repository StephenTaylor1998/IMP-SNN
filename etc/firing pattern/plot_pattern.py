import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

if_layer = neuron.IFNode(v_reset=None)

# ============================================
if_layer.reset()
T = 32
x = torch.tensor([
    1 / 12, 1 / 11, 1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6,
]) - 0.0000001

s_list = []
v_list = []
for t in range(T):
    s_list.append(if_layer(x).unsqueeze(0))
    v_list.append(if_layer.v.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)

figsize = (12, 3)
dpi = 200
visualizing.plot_2d_heatmap(
    array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
    ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)

plt.savefig(f'adjust membrane potentials.svg')
plt.show()

# ============================================
if_layer.reset()
T = 32
IMP = torch.tensor([
    0 / 12, 1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12
])

if_layer(IMP)
x = torch.ones_like(IMP) / 12
s_list = []
v_list = []
for t in range(T):
    s_list.append(if_layer(x).unsqueeze(0))
    v_list.append(if_layer.v.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)

figsize = (12, 3)
dpi = 200
visualizing.plot_2d_heatmap(
    array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
    ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)

plt.savefig(f'adjust imp.svg')
plt.show()
