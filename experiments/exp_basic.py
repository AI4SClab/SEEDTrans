import os
import torch
from model import Transformer, Informer, Autoformer, DLinear, TimesNet, PatchTST, iTransformer, CSIformer, RNN, MLWT_coiflet, MLWT_db4, MLWT_symlet, MLWT_v0, MLWT_v1, MLWT_v2


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'CSIformer': CSIformer,
            'LSTM': RNN,
            'MLWT_coiflet': MLWT_coiflet,
            'MLWT_db4': MLWT_db4,
            'MLWT_symlet': MLWT_symlet,
            'MLWT_v0': MLWT_v0,
            'MLWT_v1': MLWT_v1,
            'MLWT_v2': MLWT_v2,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
