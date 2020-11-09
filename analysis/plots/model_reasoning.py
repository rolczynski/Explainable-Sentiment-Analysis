import pathlib

import seaborn as sns
from matplotlib import pyplot as plt

HERE = pathlib.Path(__file__).parent
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_reasoning(ax, scores, max_scores):
    x = [1, 2, 3, 4]
    ax.plot(x, max_scores, color='black', linewidth=1, alpha=.5)
    ax.scatter(x, max_scores, color='black', alpha=.7,
               label='Ground Truth')
    ax.fill_between(x, max_scores, facecolor='black', alpha=.05)

    ax.plot(x, scores, color='green', linewidth=1, alpha=.5)
    ax.scatter(x, scores, color='green', alpha=.7,
               label='Basic Pattern Recognizer')
    ax.fill_between(x, scores, facecolor='green', alpha=.05)

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Thicker axes
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)

    # Give more space to have the legend in the upper-right corner.
    ax.set_ylim(bottom=0, top=.62)

    # Name axes.
    ax.set_xlabel(r'$\textbf{Number of tokens}$')
    ax.set_ylabel(r'$\textbf{PMF}$')

    # Adjust ticks.
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([1, 2, 3, '4+'])

    # Show up the legend.
    ax.legend()


def plot_confusion_matrix(ax, matrix):
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=.98, dark=0)
    sns.heatmap(matrix, ax=ax, cmap=cmap, annot=True, cbar=False, fmt='d',
                annot_kws=dict(fontsize = 16))

    ax.set_xlabel(r'\textbf{Prediction with Mask}', labelpad=20)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(r'\textbf{Prediction}', rotation=0, labelpad=40)

    ax.set_xticklabels(['Neutral', 'Negative', 'Positive'])
    ax.xaxis.tick_top()
    ax.set_yticklabels(['Neutral', 'Negative', 'Positive'], rotation=0)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.label._y = 0.475


def save_figure(max_scores, results, domain):
    results = results[-1]  # Plot only the basic pattern recognizer results.
    scores, matrix = results
    fig, ax = plt.subplots(figsize=[4.5, 3.5], dpi=600)
    plot_reasoning(ax, scores, max_scores)
    plt.tight_layout()
    plt.savefig(HERE / f'reasoning-{domain}-left.svg')

    fig, ax = plt.subplots(figsize=[4.5, 3.5], dpi=600)
    plot_confusion_matrix(ax, matrix)
    plt.tight_layout()
    plt.savefig(HERE / f'reasoning-{domain}-right.svg')
