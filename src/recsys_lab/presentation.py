from pathlib import Path
import matplotlib.pyplot as plt

def setup_style():
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'axes.axisbelow': True,
                         'grid.alpha': .22, 'font.size': 10,
                         'axes.titlesize': 12, 'figure.constrained_layout.use': True})

def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor='white', bbox_inches='tight')
    return path
