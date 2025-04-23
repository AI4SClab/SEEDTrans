import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class wavelet_decomp(nn.Module):
    """
    Wavelet decomposition block
    """

    def __init__(self, seq_len, wavelet=None):
        super(wavelet_decomp, self).__init__()
        if wavelet is None:
            #db4
            #self.h_filter = torch.Tensor([-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106])
            #self.l_filter = torch.Tensor([-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304])
            #symlet
            self.h_filter = torch.Tensor([-0.0322,-0.0126,0.0992,0.2978,-0.8037,0.49761,0.0296,-0.0757])
            self.l_filter = torch.Tensor([-0.0757,-0.0296,0.4976,0.8037,0.2978,-0.0992,-0.0126,0.0322])
            #coiflet
            #self.h_filter = torch.Tensor([-0.0232,-0.0586,-0.0953, 0.5460,-1.1494,0.5897,-0.1082,0.0841,-0.0335,0.0079,-0.0026,0.0010])
            #self.l_filter = torch.Tensor([0.0010,0.0026,0.0079,0.0335,0.0841,0.1082,0.5897,1.1494,0.5460,0.0953,0.0586,0.0232])
        else:
            w = pywt.Wavelet(wavelet)
            self.h_filter = torch.Tensor(w.dec_hi)
            self.l_filter = torch.Tensor(w.dec_lo)

        self.h_fn = nn.Linear(seq_len, seq_len)
        self.l_fn = nn.Linear(seq_len, seq_len)
        self.h_fn.weight = nn.Parameter(self.create_W(seq_len, False))
        self.l_fn.weight = nn.Parameter(self.create_W(seq_len, True))
        self.activation = nn.Sigmoid()  # sigmoid>gelu>tanh

    def forward(self, x):
        h_out = self.activation(self.h_fn(x.permute(0, 2, 1)))
        l_out = self.activation(self.l_fn(x.permute(0, 2, 1)))
        # h_out = self.pool(h_out)
        # l_out = self.pool(l_out)
        return l_out.permute(0, 2, 1), h_out.permute(0, 2, 1)

    def create_W(self, P, is_l, is_comp=False):
        if is_l:
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter
        max_epsilon = torch.min(torch.abs(filter_list))
        if is_comp:
            weight_np = torch.zeros((P, P))
        else:
            weight_np = torch.randn(P, P) * 0.1 * max_epsilon
        for i in range(0, P):
            filter_index = 0
            for j in range(i-4, P): #len(filter_list) = 8，要使分解对齐，权重矩阵需要有4个偏差
                if j<0:
                    filter_index += 1
                elif filter_index < len(filter_list):    
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return weight_np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",seq_len=96):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv11 = nn.Conv1d(in_channels=d_model*2, out_channels=d_ff, kernel_size=1)
#         self.conv12 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.decomp_hl1 = wavelet_decomp(seq_len)
#         self.decomp_hl2 = wavelet_decomp(seq_len)
#         self.conv21 = nn.Conv1d(in_channels=d_model*2, out_channels=d_ff, kernel_size=1)
#         self.conv22 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)

#         l1,h1=self.decomp_hl1(x)
#         # print('* x l1 ', x.shape,l1.shape)
#         xh1 = torch.cat([x,h1],dim=-1)
#         xh1 = self.dropout(self.activation(self.conv11(xh1.transpose(-1, 1))))
#         xh1 = self.dropout(self.conv12(xh1).transpose(-1, 1))
#         x= self.norm1(x+xh1)

#         l2,h2=self.decomp_hl2(l1)
#         xh2 = torch.cat([x,h2],dim=-1)
#         xh2 = self.dropout(self.activation(self.conv21(xh2.transpose(-1, 1))))
#         xh2 = self.dropout(self.conv22(xh2).transpose(-1, 1))
#         y = self.norm2(x + xh2 + l2)

#         return y, attn


# class Encoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer

#     def forward(self, x, attn_mask=None):
#         # x [B, L, D]
#         attns = []
#         if self.conv_layers is not None:
#             for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
#                 x, attn = attn_layer(x, attn_mask=attn_mask)
#                 x = conv_layer(x)
#                 attns.append(attn)
#             x, attn = self.attn_layers[-1](x)
#             attns.append(attn)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, attn = attn_layer(x, attn_mask=attn_mask)
#                 attns.append(attn)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, attns

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu", seq_len=96):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        ## v1.0
        self.attention = attention

        self.conv1 = nn.Conv1d(in_channels=d_model * 3, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        ### v3.1
        self.conv3 = nn.Conv1d(in_channels=d_model * 4, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp_ts1 = series_decomp(moving_avg)
        self.decomp_ts2 = series_decomp(moving_avg)
        self.decomp_hl1 = wavelet_decomp(seq_len)
        self.decomp_hl2 = wavelet_decomp(seq_len)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    ## v1.0 lol
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x_ts = x + self.dropout(new_x)
        x_s, _ = self.decomp_ts1(x_ts)
        # print('* en x ',x.shape)
        x_l, x_h = self.decomp_hl1(x)
        y = torch.cat([x_s, x_h, x_l], dim=-1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res_s, _ = self.decomp_ts2(x_s + y)
        res_l, res_h = self.decomp_hl2(x_l + y)  # v1.1: x_h
        res_s = torch.cat([res_s, x_h, res_h, res_l], dim=-1)  # [32, 144, 64]
        res_s = self.dropout(self.conv3(res_s.transpose(-1, 1))).transpose(-1, 1)  # [32, 144, 16]
        return res_s, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", seq_len=96):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv21 = nn.Conv1d(in_channels=d_model * 3, out_channels=d_ff, kernel_size=1)
        self.conv22 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv31 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv32 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp_hl1 = wavelet_decomp(seq_len)
        self.decomp_hl2 = wavelet_decomp(seq_len)
        self.decomp_hl3 = wavelet_decomp(seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        l1, h1 = self.decomp_hl1(x)
        l2, h2 = self.decomp_hl2(l1)
        xh2 = torch.cat([x, h1, h2], dim=-1)
        xh2 = self.dropout(self.activation(self.conv21(xh2.transpose(-1, 1))))
        xh2 = self.dropout(self.conv22(xh2).transpose(-1, 1))
        x = self.norm2(x + xh2)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        l3, h3 = self.decomp_hl3(l2)
        y = self.norm3(x + h3 + l3)
        return y


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

# ## v3.0
# class DecoderLayer(nn.Module):
#     """
#     Autoformer decoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
#                  moving_avg=25, dropout=0.1, activation="relu",seq_len=96):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model*4, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         # self.conv3 = nn.Conv1d(in_channels=d_model*2, out_channels=d_model, kernel_size=1, bias=False)
#         # self.conv4=nn.Conv1d(in_channels=d_model*2, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp_ts1 = series_decomp(moving_avg)
#         self.decomp_ts2 = series_decomp(moving_avg)
#         self.decomp_ts3 = series_decomp(moving_avg)
#         self.decomp_hl1 = wavelet_decomp(seq_len)
#         self.decomp_hl2 = wavelet_decomp(seq_len)
#         self.decomp_hl3 = wavelet_decomp(seq_len)
#         self.dropout = nn.Dropout(dropout)
#         self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
#                                     padding_mode='circular', bias=False)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         x_ts = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x_s, trend1 = self.decomp_ts1(x_ts)
#         # print('* x ',x.shape)
#         x_l1,x_h1=self.decomp_hl1(x)
#         x_s = x_s + self.dropout(self.cross_attention(
#             x_s, cross, cross,
#             attn_mask=cross_mask
#         )[0])

#         x_s, trend2 = self.decomp_ts2(x_s)
#         x_l2,x_h2=self.decomp_hl2(x_l1)

#         y = torch.cat([x_s,x_h1,x_h2,x_l2],dim=-1)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         x_s, trend3 = self.decomp_ts3(x_s + y)
#         x_l3,x_h3=self.decomp_hl3(x_l2+y)
#         # x_s= self.dropout(self.conv3(torch.cat([x_s,x_h3],dim=-1).transpose(-1,1))).transpose(-1,1)
#         # v3.1
#         residual_trend = trend1 + trend2 + trend3 + x_h3 + x_l3
#         # residual_trend=self.dropout(self.conv4(torch.cat([residual_trend,x_l3],dim=-1).transpose(-1,1))).transpose(-1,1)
#         # print('* x_s x_h3',x_s.shape,residual_trend.shape)
#         residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
#         return x_s, residual_trend

# class Decoder(nn.Module):
#     """
#     Autoformer encoder
#     """
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super(Decoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection

#     def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
#         for layer in self.layers:
#             x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
#             trend = trend + residual_trend

#         if self.norm is not None:
#             x = self.norm(x)

#         if self.projection is not None:
#             x = self.projection(x)
#         return x, trend