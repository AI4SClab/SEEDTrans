import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np

class Model(nn.Module):
    """
    LSTM
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        bidirectional=False
        if 'Bi' in configs.model:
            bidirectional=True
        if 'LSTM' in configs.model:
            rnn=nn.LSTM
        if 'GRU' in configs.model:
            rnn=nn.GRU
        hidden_size,num_layers=512,3
        self.rnn = rnn(input_size=configs.d_model,hidden_size=hidden_size,num_layers=num_layers, bidirectional=bidirectional,batch_first=True) # 12
        self.projection=nn.Linear(hidden_size, configs.c_out, bias=True)

    def forward(self, x, x_mark, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # print('* xm x ',x_mark.shape,x.shape)
        # x = torch.cat([x_mark,x],dim=-1)
        # print('* x ',x.shape)

        x= self.enc_embedding(x, x_mark)

        x,state= self.rnn(x)
        x = self.projection(x)
        return x[:, -self.pred_len:, :]  # [B, L, D]
