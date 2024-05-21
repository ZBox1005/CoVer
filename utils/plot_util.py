
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F

def plot_distribution(args, id_scores, ood_scores, out_dataset):
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')

def plot_aug_distribution(id_scores, aug_scores, aug_method):
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    sns.displot({"ID": -1 * id_scores, f"{aug_method}": -1 * aug_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    plt.savefig(os.path.join(f"visual/ImageNet/2/{aug_method}.png"), bbox_inches='tight')

def plot_all_aug_distribution(id_scores, all_aug_scores):
    sns.set(style="white", palette="muted")
    palette1 = '#A8BAE3'
    palette2 = '#55AB83'
    n_charts = len(all_aug_scores)
    n_cols = 5
    n_rows = n_charts // n_cols + (n_charts % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(100, 20 * n_rows))

    for ax, (name, aug_scores) in zip(axes.flatten(), all_aug_scores.items()):
        # 使用seaborn绘制KDE图
        sns.kdeplot(-1 * id_scores, label="ID", fill=True, alpha=0.8, palette=palette1, ax=ax)
        sns.kdeplot(-1 * aug_scores, label="AUG", fill=True, alpha=0.8, palette=palette2, ax=ax)
        ax.set_title(name)
        ax.legend()

    # 隐藏多余的子图
    for i in range(n_charts, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

    # plt.tight_layout()
    plt.savefig(os.path.join(f"visual/ImageNet/2/all_aug_methods.png"), bbox_inches='tight')

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9) 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


