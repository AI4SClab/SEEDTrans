import pywt
import matplotlib.pyplot as plt

wavelets = ['db4', 'coif4', 'sym4']
titles = ['Daubechies 4', 'Coiflet 4', 'Symlet 4']

# 设置画布
spine_alpha = 0.3
fig, axes = plt.subplots(1, 3, figsize=(8, 5))

# 遍历每种小波
for i, wavelet_name in enumerate(wavelets):
    wavelet = pywt.Wavelet(wavelet_name)

    # 得到尺度函数（phi）和小波函数（psi）
    phi, psi, x = wavelet.wavefun(level=10)

    ax = axes[i]
    line1, = ax.plot(x, phi, label='Scaling Function', color='#8DB4E2')
    line2, = ax.plot(x, psi, label='Wavelet Function', color='#FFE599')
    ax.set_title(titles[i], fontsize=16)
    ax.set_xlabel('x',fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_alpha(spine_alpha)

# 设置全局图例在下方
fig.legend([line1, line2], ['Scaling Function', 'Wavelet Function'],
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.05),fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25, top=0.85)
plt.savefig('./wavelet_functions.pdf', format='pdf', bbox_inches='tight')
plt.show()
