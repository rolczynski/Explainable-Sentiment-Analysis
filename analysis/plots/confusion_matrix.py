import pathlib

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

HERE = pathlib.Path(__file__).parent
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot(ax, matrix, name):
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=.98, dark=0)
    sns.heatmap(matrix, ax=ax, cmap=cmap, annot=True, cbar=False, fmt='d')
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_yticklabels([])
    ax.set_xlabel(name)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')


def save_figure(results, domain, names=('Random', 'Attention', 'Gradient', 'Basic')):
    y = [is_change for acc, matrix, is_change in results]
    matrices = [confusion_matrix(y[i], y[j]) for i, j in [(3, 0), (3, 1), (3, 2)]]
    fig, axes = plt.subplots(1, 3, figsize=[5.7, 2.2], dpi=600)
    x_labels, y_label = names[:-1], names[-1]
    for ax, matrix, name in zip(axes, matrices, x_labels):
        plot(ax, matrix, name)
    axes[0].set_ylabel(y_label)
    axes[0].set_yticklabels(['No', 'Yes'])
    plt.tight_layout(pad=2., w_pad=3.)
    plt.savefig(HERE / f'confusion-matrix-{domain}.svg')
