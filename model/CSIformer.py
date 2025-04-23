import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.CSIformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_Linear
from layers.MaskModel import MaskModel
from layers.MSPI import MSPINet


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        if configs.freq == 'h':
            self.x_mark_dim = 4
        elif configs.freq == 't':
            self.x_mark_dim = 5
        elif configs.freq == 's':
            self.x_mark_dim = 6

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        mask_size = configs.dec_in + self.x_mark_dim

        # ChannelTune Masking
        self.mask = MaskModel(input_shape=(configs.batch_size, mask_size, configs.seq_len))

        # Multi-Scale Pyramid Integration
        self.mspi = MSPINet(configs.mspi_layers, configs.pool_type)

        # Embedding
        self.embedding = DataEmbedding_Linear(configs.seq_len, configs.mspi_layers, configs.d_model,
                                               configs.embed, configs.freq, configs.dropout)

        # Encoder with Multi-Scale Attention Fusion
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Projection
        self.mapping = nn.Linear(configs.d_model, configs.map_dim, bias=True)
        self.projection = nn.Linear(configs.map_dim, configs.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, C = x_enc.shape  # B L C
        # B: batch_size
        # L: seq_len
        # C: Channels

        # ChannelTune Masking
        x = torch.cat([x_enc, x_mark_enc], 2)
        x_inv = x.permute(0, 2, 1)
        y = self.mask(x_inv)

        # Multi-Scale Pyramid Integration
        y_mspi = self.mspi(y)

        # Embedding
        y_mspi, y = self.embedding(y_mspi, y)

        # Encoder with Multi-Scale Attention Fusion
        z, _ = self.encoder(y_mspi, y, attn_mask=None)
        
        # Projection
        dec_out = self.mapping(z)
        dec_out = nn.functional.gelu(dec_out)
        dec_out = self.projection(dec_out).permute(0, 2, 1)[:, :, :C]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]  # [B, N, C]
