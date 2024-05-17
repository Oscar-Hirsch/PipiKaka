# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import weight_scaling_init


# Transformer (encoder) https://github.com/jadore801120/attention-is-all-you-need-pytorch
# Original Copyright 2017 Victor Huang
#  MIT License (https://opensource.org/licenses/MIT)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''


    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) 
        self.w_2 = nn.Linear(d_hid, d_in) 
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

# positional encoding is defined but not used. 
# It is skipped in the initialization of the Transformer encoder.
class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, height, width):
        self.dimensions = d_hid
        self.height = height
        self.width = width
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(height, width, d_hid))

    def _get_sinusoid_encoding_table(self, height, width, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table_h = np.array([get_position_angle_vec(pos_i) for pos_i in range(height)])
        sinusoid_table_w = np.array([get_position_angle_vec(pos_i) for pos_i in range(width)])

        sinusoid_table_h[:, 0::2] = np.sin(sinusoid_table_h[:, 0::2])
        sinusoid_table_h[:, 1::2] = np.cos(sinusoid_table_h[:, 1::2])
        sinusoid_table_w[:, 0::2] = np.sin(sinusoid_table_w[:, 0::2])
        sinusoid_table_w[:, 1::2] = np.cos(sinusoid_table_w[:, 1::2])

        sinusoid_table_h = torch.FloatTensor(sinusoid_table_h).unsqueeze(0).unsqueeze(2)
        sinusoid_table_w = torch.FloatTensor(sinusoid_table_w).unsqueeze(0).unsqueeze(1)

        sinusoid_table = sinusoid_table_h + sinusoid_table_w
        return sinusoid_table

    def forward(self, x):
        
        pos_table_flat = self.pos_table.view(1, self.height * self.width, self.dimensions)
        return x + pos_table_flat[:, :x.size(1), :].clone().detach()



# defines the encoder layers in the transformers
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, spec_width=0, spec_height=0, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1, scale_emb=False):

        super().__init__()
        # the positional encoding is skipped here
        if spec_height > 0:
            self.position_enc = PositionalEncoding(d_model, height=spec_height, width=spec_width)
        else:
            self.position_enc = lambda x: x
    
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []
        B, C, H, W = src_seq.size()

        # reshape for transformer encoder
        src_seq_reshaped = src_seq.view(B, C, H * W).transpose(1, 2)  # B, H*W, C

        # dropout, and layer normalization
        if self.scale_emb:
            src_seq_reshaped *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(src_seq_reshaped))
        enc_output = self.layer_norm(enc_output)

        # encoder layers
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # reshape the tensor back to its original shape
        enc_output_reshaped = enc_output.transpose(1, 2).view(B, C, H, W)
        
        # this is set to false so it wont be called
        if return_attns:
            return enc_output_reshaped, enc_slf_attn_list
        
        return enc_output_reshaped


# CleanUNet architecture

# this pads the time dimension of the spectrogram on the right side
# the function calculates the length of the time dimension after the downsampling operations and subsequently length of the same dimension after the upsampling process.
# from the final value we substract the orinal shape so we obtain the amount we need to pad

def padding_time(x, D, K, S):
    """padding zeroes to the right side of the time dimension"""

    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)

    for _ in range(D):
        L = (L - 1) * S + K
    
    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x

# this pads the frequency dimension of the spectrogram on the bottom
# the function calculates the length of the frequency dimension after the downsampling operations and subsequently the length of the same dimension after the upsampling process.
# from the final value we substract the orinal shape so we obtain the amount we need to pad

def padding_freq(x, D, K, S):
    """padding zeroes to the rigth side of the frequency dimension"""
    
    L = x.shape[-2]

    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)

    for _ in range(D):
        L = (L - 1) * S + K

    L = int(L)

    x = F.pad(x, (0, 0, 0, L - x.shape[-2]))

    return x

# defines the CleanUNet architecture
class CleanUNet(nn.Module):
    """ CleanUNet architecture. """

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=3, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048):
        
        """
        Parameters:
        channels_input (int):   input channels
        channels_output (int):  output channels
        channels_H (int):       middle channels H that controls capacity
        max_H (int):            maximum H
        encoder_n_layers (int): number of encoder/decoder layers D
        kernel_size (int):      kernel size K
        stride (int):           stride S
        tsfm_n_layers (int):    number of self attention blocks N
        tsfm_n_head (int):      number of heads in each self attention block
        tsfm_d_model (int):     d_model of self attention
        tsfm_d_inner (int):     d_inner of self attention
        """

        super(CleanUNet, self).__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        # encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # here we changed the Conv1D from the WBM to Conv2d for the SBM to handle the two dimensions present in the spectrogram.
        # Similarly we change the ConvTranspose1d to ConvTranspose2d operations for the same purpose

        for i in range(encoder_n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(channels_input, channels_H, kernel_size, stride),
                nn.ReLU(),
                nn.Conv2d(channels_H, channels_H * 2, 1), 
                nn.GLU(dim=1)
            ))
            channels_input = channels_H

            if i == 0:
                self.decoder.append(nn.Sequential(
                    nn.Conv2d(channels_H, channels_H * 2, 1), 
                    nn.GLU(dim=1),
                    nn.ConvTranspose2d(channels_H, channels_output, kernel_size, stride)
                ))
            else:
                self.decoder.insert(0, nn.Sequential(
                    nn.Conv2d(channels_H, channels_H * 2, 1), 
                    nn.GLU(dim=1),
                    nn.ConvTranspose2d(channels_H, channels_output, kernel_size, stride),
                    nn.ReLU()
                ))
            channels_output = channels_H
            
            # double H but keep below max_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)
        
        # self attention block
        self.tsfm_conv1 = nn.Conv2d(channels_output, tsfm_d_model, kernel_size=1)
        
        # the transformer is initialized here. the spec_height is set to 0 so that the positional encoding is skipped 
        self.tsfm_encoder = TransformerEncoder(n_layers=tsfm_n_layers, 
                                               n_head=tsfm_n_head, 
                                               d_k=tsfm_d_model // tsfm_n_head, 
                                               d_v=tsfm_d_model // tsfm_n_head, 
                                               d_model=tsfm_d_model, 
                                               d_inner=tsfm_d_inner, 
                                               dropout=0.0,   
                                               scale_emb=False,
                                               spec_height = 0,
                                               spec_width = 0)
        
        self.tsfm_conv2 = nn.Conv2d(tsfm_d_model, channels_output, kernel_size=1)

        # weight scaling initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                weight_scaling_init(layer)

    def forward(self, noisy_audio):
        # (B, H, W) -> (B, C, H, W)
        if len(noisy_audio.shape) == 3:
            noisy_audio = noisy_audio.unsqueeze(1)

        B, C, H, W = noisy_audio.shape
        assert C == 1

        # normalization and padding
        std = noisy_audio.std(dim=[2,3], keepdim=True) + 1e-3
        noisy_audio /= std
        
        # we apply the padding on the frequency and time dimension
        x = padding_time(noisy_audio, self.encoder_n_layers, self.kernel_size, self.stride)
        x = padding_freq(x, self.encoder_n_layers, self.kernel_size, self.stride)
        
        # encoder
        skip_connections = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]

        # calculate attention mask for transformer encoder
        len_s = x.shape[-1]  
        width_s = x.shape[-2]
        attn_mask = (1 - torch.triu(torch.ones((1, len_s*width_s, len_s*width_s), device=x.device), diagonal=1)).bool()

        # transformer encoder
        x = self.tsfm_conv1(x)
        transformer_output = self.tsfm_encoder(x, src_mask=attn_mask)
       
       # get already reshaped tensor (tensor is reshaped in the TransformerEncoder Class)
        x = transformer_output
        x = self.tsfm_conv2(x)

        # decoder
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x = x + skip_i[:, :, :x.shape[-2], :x.shape[-1]]
            x = upsampling_block(x)
        
        # applying ReLU to ensure only non-negative values
        x = F.relu(x)  
        x = x[:, :, :H, :W] * std

        return x



if __name__ == '__main__':
    import json
    import argparse 
    import os
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/DNS-large-full.json', 
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    network_config = config["network_config"]

    model = CleanUNet(**network_config).cuda()
    from util import print_size
    print_size(model, keyword="tsfm")
    
    input_data = torch.ones([4,1,401, 501]).cuda()
    output = model(input_data)
    print(output.shape)

    y = torch.rand([4,1,401, 501]).cuda()
    loss = torch.nn.MSELoss()(y, output)
    loss.backward()
    print(loss.item())
    