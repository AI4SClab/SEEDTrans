import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.MLWT_EncDec_db4 import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, my_Layernorm,series_decomp
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.conv1 = nn.Conv1d(in_channels=configs.d_model*2, out_channels=configs.d_model, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=configs.d_model, kernel_size=1, bias=False)

        # # Decomp
        # kernel_size = configs.moving_avg
        # self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        #self.enc_embedding = DataEmbedding(configs.enc_in - 3, configs.d_model, configs.embed, configs.freq,
        #                                   configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        ####test3.9
        self.linear = nn.Linear(3, configs.d_model)
        self.linear2 = nn.Linear(configs.d_model, 256)
        self.linear3 = nn.Linear(configs.d_model * 3, 256)
        self.linear4 = nn.Linear(256, 3)
        self.linear5 = nn.Linear(configs.enc_in, 64)
        self.linear6 = nn.Linear(64, configs.enc_in)
        self.selected_columns = nn.Parameter(torch.zeros([1, configs.enc_in], dtype=torch.float32))
        self.selected_columns2 = nn.Parameter(torch.ones([1, configs.enc_in], dtype=torch.float32))
        

        # # Encoder
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #             seq_len=configs.seq_len
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer( #AutoCorrelationLayer
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, #AutoCorrelation    
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    seq_len=configs.seq_len
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        ## Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    seq_len=configs.label_len+configs.pred_len
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    #3.9增加if判断，对于特定数据集才用这个，其他数据集用原来的forward
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #print(self.selected_columns)
        ####test3.9 x_humid
        sharpen = 10
        threshold = torch.mean(self.selected_columns)
        #selected_columns = torch.sigmoid((self.selected_columns - threshold) * sharpen)
        selected_columns = torch.sigmoid(self.selected_columns)  # 使用 sigmoid 将值映射到 (0, 1)
        x_enc = torch.mul(x_enc, selected_columns)
        _, top_columns = torch.topk(selected_columns, 6)#6
        x_a = x_enc[:, :, top_columns[0][:3]]                                   #x_hum[32, 144, 3]
        selected_columns = selected_columns[:, top_columns[0]]
        #print("top_columns_a",top_columns[0][:3])
        #print("top_columns_b",top_columns[0][3:])
        weights = selected_columns.squeeze().tolist()  # [num_variables]

        # 将 topk 索引转为列表
        topk_indices = top_columns[0].tolist()

        # 写入日志
        with open("logs/topk_variable_db4.txt", "a") as f:
            f.write("TopK Variables: " + ",".join(map(str, topk_indices)) + "\n")
            f.write("All Weights: " + ",".join(f"{w:.4f}" for w in weights) + "\n")
            f.write("\n")
        #with open("logs/topk_variable_db4.txt", "a") as f:
            #f.write(",".join(map(str, top_columns[0].tolist())) + "\n")
        x_b = x_enc[:, :, top_columns[0][3:]]                                   # x_hum[32, 144, 3]
        ####test3.9
        out_hum = self.linear(x_a)                                        #out_hum[32, 144, 16]
        out_hum = self.linear2(out_hum)                                     # out_hum[32, 144, 256]
        out_hum = self.conv2(out_hum.transpose(-1, 1)).transpose(-1, 1)     # out_hum[32, 144, 16]
        out_cld = self.linear(x_b)  # out_hum[32, 144, 16]
        out_cld = self.linear2(out_cld)  # out_hum[32, 144, 256]
        out_cld = self.conv2(out_cld.transpose(-1, 1)).transpose(-1, 1)  # out_hum[32, 144, 16]

        enc_out = self.enc_embedding(x_enc, x_mark_enc)                     #enc_out[32, 144, 16]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)     #enc_out[32, 144, 16]
        out_mix = torch.cat([enc_out, out_hum, out_cld], dim=-1)                     #out_mix[32, 144, 48]
        out_mix = self.linear3(out_mix)                                     #out_mix[32, 144, 256]
        weight = nn.functional.relu(out_mix)                                #weight 32 x 144 x 256
        weight = self.linear4(weight)                                       #weight 32 x 144 x 2
        weight = nn.functional.softmax(weight, dim=2)                       #weight 32 x 144 x 2
        mean_weight = torch.mean(weight, dim=0)  
        mean_weight = torch.mean(mean_weight, dim=0)  
        #print(mean_weight)
        # Apply weight for enc_out
        enc_out = enc_out * weight[:, :, 0:1] + out_hum * weight[:, :, 1:2] + out_cld * weight[:, :, 2:3] #enc_out[32, 144, 16]


        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]