# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 24, 1)

sunny_actual = np.array([0,0,0,0.033,0.1,0.33,0.67,1.2,1.73,2.15,2.48,2.74,2.83,2.8,2.75,2.68,2.48,1.86,1.03,0.53,0.26,0,0,0])
sunny_SEEDTrans = np.array([0.0, 0.0, 0.0, 0.036, 0.1, 0.24, 0.556, 0.996, 1.543, 2.06, 2.402, 2.651, 2.781, 2.826, 2.757, 2.607, 2.454, 1.9, 1.122, 0.599, 0.249, 0.0, 0.0, 0.0])
sunny_iTransformer =np.array([0.0, 0.0, 0.0, 0.0, 0.042, 0.12, 0.447,  0.816, 1.4, 1.866, 2.173, 2.301, 2.614,  2.791, 2.536, 2.404, 2.289, 1.619, 1.516, 0.785, 0.388, 0.15, 0.005, 0.0])
sunny_CNN-BiLSTM = np.array([0, 0, 0, 0, 0.04, 0.13, 0.366, 0.733, 1.03, 1.533, 1.933, 2.245, 2.55, 2.633, 2.773, 2.533, 2.366, 1.833, 1.467, 0.867, 0.433, 0.1, 0.003, 0])


overcast_actual = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.033, 0.117, 0.183, 0.243, 0.333, 0.408, 0.491, 0.585, 0.696, 0.683, 0.628, 0.538, 0.383, 0.133, 0.05, 0.0, 0.0, 0.0, 0.0])
overcast_SEEDTrans = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.035, 0.122, 0.175, 0.263, 0.321, 0.395, 0.478, 0.576, 0.687, 0.685, 0.633, 0.548, 0.395, 0.196, 0.07, 0.0, 0.0, 0.0, 0.0])
overcast_iTransformer = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.11, 0.249, 0.38, 0.462, 0.539, 0.666, 0.707, 0.68, 0.538, 0.392, 0.284, 0.163, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
overcast_CNN-BiLSTM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.12, 0.28, 0.39, 0.44, 0.57, 0.59, 0.66, 0.7, 0.67, 0.63, 0.52, 0.43, 0.29, 0.14, 0.05, 0.0, 0.0, 0.0])


cloudy_actual = np.array([0.0, 0.0, 0.0, 0.0, 0.006, 0.14, 0.266, 0.322, 0.457, 0.6, 0.7431, 0.864, 0.917, 0.315, 1.073, 0.843, 0.774, 0.567, 0.343, 0.033, 0.0, 0.0, 0.0, 0.0])
cloudy_SEEDTrans = np.array([0, 0, 0, 0, 0.003, 0.11, 0.25, 0.31, 0.444, 0.588, 0.715, 0.848, 0.901, 0.4, 0.997, 0.864, 0.71, 0.527, 0.355, 0.03, 0, 0, 0, 0])
cloudy_iTransformer = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.17, 0.22, 0.39, 0.44, 0.54, 0.66, 0.92, 0.77, 0.89, 0.74, 0.66, 0.45, 0.31, 0.03, 0.0, 0.0, 0.0, 0.0])
cloudy_CNN-BiLSTM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.1, 0.21, 0.39, 0.45, 0.63, 0.76, 0.82, 0.68, 0.95, 0.79, 0.67, 0.46, 0.23, 0.03, 0.0, 0.0, 0.0, 0.0])



rainy_actual = np.array([0, 0, 0, 0, 0, 0, 0.03, 0.08, 0.13, 0.18, 0.22, 0.28, 0.33, 0.43,0.39 , 0.31, 0.2, 0.15, 0.06, 0, 0, 0, 0, 0])
rainy_SEEDTrans = np.array([0, 0, 0, 0, 0, 0, 0.015, 0.0689, 0.1158, 0.1699, 0.2105, 0.2618, 0.322, 0.427,0.386 , 0.297, 0.188, 0.14, 0.07, 0, 0, 0, 0, 0])
rainy_iTransformer = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.017, 0.056, 0.088, 0.129, 0.155, 0.196, 0.232, 0.353, 0.333, 0.257, 0.139, 0.082, 0.028, 0.0, 0.0, 0.0, 0.0, 0.0])
rainy_CNN-BiLSTM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.064, 0.086,  0.157, 0.199, 0.273, 0.307, 0.379, 0.316, 0.225, 0.138, 0.094, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0])

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(time, sunny_actual, label='Actual', color='black', marker='o', linestyle='-')
axs[0, 0].plot(time, sunny_iTransformer, label='iTransformer', color='green', marker='o', linestyle='-')
axs[0, 0].plot(time, sunny_CNN-BiLSTM, label='CNN-BiLSTM', color='red', marker='o', linestyle='-')
axs[0, 0].plot(time, sunny_SEEDTrans, label='SEEDTrans', color='blue', marker='o', linestyle='-')
axs[0, 0].set_title('Sunny Day',fontsize=16)
axs[0, 0].set_xlabel('Time (Hour)',fontsize=14)
axs[0, 0].set_ylabel('Power',fontsize=14)
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle='--', alpha=0.6)
axs[0, 0].set_ylim(0, 3)

axs[0, 1].plot(time, overcast_actual, label='Actual', color='black', marker='o', linestyle='-')
axs[0, 1].plot(time, overcast_iTransformer, label='iTransformer', color='green', marker='o', linestyle='-')
axs[0, 1].plot(time, overcast_CNN-BiLSTM, label='CNN-BiLSTM', color='red', marker='o', linestyle='-')
axs[0, 1].plot(time, overcast_SEEDTrans, label='SEEDTrans', color='blue', marker='o', linestyle='-')
axs[0, 1].set_title('Overcast Day',fontsize=16)
axs[0, 1].set_xlabel('Time (Hour)',fontsize=14)
axs[0, 1].set_ylabel('Power',fontsize=14)
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle='--', alpha=0.6)
axs[0, 1].set_ylim(0, 3)

axs[1, 0].plot(time, cloudy_actual, label='Actual', color='black', marker='o', linestyle='-')
axs[1, 0].plot(time, cloudy_iTransformer, label='iTransformer', color='green', marker='o', linestyle='-')
axs[1, 0].plot(time, cloudy_CNN-BiLSTM, label='CNN-BiLSTM', color='red', marker='o', linestyle='-')
axs[1, 0].plot(time, cloudy_SEEDTrans, label='SEEDTrans', color='blue', marker='o', linestyle='-')
axs[1, 0].set_title('Cloudy Day',fontsize=16)
axs[1, 0].set_xlabel('Time (Hour)',fontsize=14)
axs[1, 0].set_ylabel('Power',fontsize=14)
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle='--', alpha=0.6)
axs[1, 0].set_ylim(0, 3)

axs[1, 1].plot(time, rainy_actual, label='Actual', color='black', marker='o', linestyle='-')
axs[1, 1].plot(time, rainy_iTransformer, label='iTransformer', color='green', marker='o', linestyle='-')
axs[1, 1].plot(time, rainy_CNN-BiLSTM, label='CNN-BiLSTM', color='red', marker='o', linestyle='-')
axs[1, 1].plot(time, rainy_SEEDTrans, label='SEEDTrans', color='blue', marker='o', linestyle='-')
axs[1, 1].set_title('Rainy Day',fontsize=16)
axs[1, 1].set_xlabel('Time (Hour)',fontsize=14)
axs[1, 1].set_ylabel('Power',fontsize=14)
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle='--', alpha=0.6)
axs[1, 1].set_ylim(0, 3)
plt.savefig('./weather.pdf', format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()