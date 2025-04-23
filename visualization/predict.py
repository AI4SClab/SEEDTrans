import numpy as np
import matplotlib.pyplot as plt


def visual(true, preds=None, name=''):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.savefig(name, bbox_inches='tight', format='png')
    plt.show()
preds = np.load('./results/PV0_128_64_ARIMA_PV_ftMS_sl128_pl128_Exp/pred.npy')
trues = np.load('./results/PV0_128_64_ARIMA_PV_ftMS_sl128_pl128_Exp/true.npy')
pred_640 = preds[20]
visual(trues[20], pred_640, name='./pic/ARIMA.png')

