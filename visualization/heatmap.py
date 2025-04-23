# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_high = pd.read_csv('./visualization/high_origin.csv', index_col=0)
data_low = pd.read_csv('./visualization/low.csv', index_col=0)

fig, axes = plt.subplots(nrows=2, ncols=1, dpi=400)

# 低成分热力图
heatmap_low = axes[0].imshow(data_low.T, cmap='Reds', interpolation='nearest')
axes[0].set_xticks([])
axes[0].set_yticks(range(len(data_low.columns)))
axes[0].set_yticklabels(data_low.columns, fontsize=8)
for i in range(len(data_low.columns)):
    for j in range(len(data_low.index)):
        text_color = 'white' if data_low.iloc[j, i] > 0.3 else 'black'
        axes[0].text(j, i, f'{data_low.iloc[j, i]:.2f}', ha='center', va='center', color=text_color, fontsize=7)
axes[0].set_xlabel('(a)', fontsize=8, labelpad=8)
axes[0].set_ylabel('Low Component', fontsize=8, labelpad=15)
axes[0].tick_params(axis='both', pad=3)
axes[0].set_xticks(np.arange(data_low.shape[0]+1)-0.5, minor=True)
axes[0].set_yticks(np.arange(data_low.shape[1]+1)-0.5, minor=True)
axes[0].grid(which='minor', color='white', linewidth=0.5, linestyle='-')


# 高成分热力图
heatmap_high = axes[1].imshow(data_high.T, cmap='Reds', interpolation='nearest')
axes[1].set_xticks(range(len(data_high.index)))
axes[1].set_xticklabels(data_high.index, fontsize=8, rotation=45, ha='right', rotation_mode='anchor')
axes[1].set_yticks(range(len(data_high.columns)))
axes[1].set_yticklabels(data_high.columns, fontsize=8)
for i in range(len(data_high.columns)):
    for j in range(len(data_high.index)):
        text_color = 'white' if data_high.iloc[j, i] > 0.3 else 'black'
        axes[1].text(j, i, f'{data_high.iloc[j, i]:.2f}', ha='center', va='center', color=text_color, fontsize=7)
axes[1].set_xlabel('(b)', fontsize=8, labelpad=8)
axes[1].set_ylabel('High Component', fontsize=8, labelpad=15)
axes[1].tick_params(axis='both', pad=3)
axes[1].set_xticks(np.arange(data_high.shape[0]+1)-0.5, minor=True)
axes[1].set_yticks(np.arange(data_high.shape[1]+1)-0.5, minor=True)
axes[1].grid(which='minor', color='white', linewidth=0.5, linestyle='-')

# 添加颜色条并调整
colorbar_low = plt.colorbar(heatmap_low, ax=axes[0], fraction=0.03)
colorbar_low.ax.tick_params(labelsize=6)
colorbar_high = plt.colorbar(heatmap_high, ax=axes[1], fraction=0.03)
colorbar_high.ax.tick_params(labelsize=6)

plt.subplots_adjust(hspace=0.1)
plt.savefig('./heatmap.pdf', format='pdf', bbox_inches='tight')
plt.show()
