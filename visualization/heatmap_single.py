# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('./visualization/high.csv', index_col=0)

fig, ax = plt.subplots(dpi=400)

heatmap = ax.imshow(data.T, cmap='Reds', interpolation='nearest')

ax.set_xticks(range(len(data.index)))
ax.set_xticklabels(data.index, fontsize=8, rotation=45, ha='right', rotation_mode='anchor')
ax.set_yticks(range(len(data.columns)))
ax.set_yticklabels(data.columns, fontsize=8)

for i in range(len(data.columns)):
    for j in range(len(data.index)):
        color = 'white' if data.iloc[j, i] > 0.3 else 'black'
        ax.text(j, i, f'{data.iloc[j, i]:.2f}', ha='center', va='center', color=color, fontsize=7)

ax.set_ylabel('Different Component', fontsize=8, labelpad=15)

ax.tick_params(axis='x', pad=3)
ax.tick_params(axis='y', pad=4)
ax.set_xticks(np.arange(data.shape[0]+1)-0.5, minor=True)
ax.set_yticks(np.arange(data.shape[1]+1)-0.5, minor=True)
ax.grid(which='minor', color='white', linewidth=0.5, linestyle='-')

colorbar = plt.colorbar(heatmap, ax=ax, fraction=0.02, pad=0.04)
colorbar.ax.tick_params(labelsize=6)

plt.savefig('./heatmap_single.pdf', format='pdf', bbox_inches='tight')

plt.show()
