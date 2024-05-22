import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt
from spikingjelly.activation_based.functional import reset_net
from spikingjelly.activation_based.neuron import IFNode


def pattern(number):
    return np.array(list(f'{number:04b}'), dtype=int)


def pattern_index(np_array):
    return int(''.join(np_array.astype(str)), 2)


def get_all_pattern(length):
    return np.array([pattern(i) for i in range(length)])


def pattern_mapping(matrix, weight, cover_value=False, imp=0.0):
    length = matrix.shape[0]
    x = torch.tensor(get_all_pattern(length)).transpose(0, 1)
    # model = IFNode(step_mode='m', v_reset=0.0)
    model = IFNode(step_mode='m', v_reset=None)
    reset_net(model)
    model.v = imp
    with torch.no_grad():
        y = model(x * weight)

    index = [pattern_index(p.long().numpy()) for p in y.transpose(0, 1)]

    for i1, i2 in zip(index, range(length)):
        if cover_value or matrix[i1, i2] == 0:
            matrix[i1, i2] = 1
    return matrix


if __name__ == '__main__':
    matrix = np.zeros((16, 16))

    cover_value = True
    imp = 0.00

    for weight in np.linspace(0, 1, num=16 + 1):
        matrix = pattern_mapping(matrix, weight, True, imp)

    print(np.sum(matrix) / 16)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    seaborn.heatmap(matrix, cbar=False)
    ax.xaxis.tick_top()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.93)
    plt.savefig(f'IMP={imp}.svg')
    plt.show()
