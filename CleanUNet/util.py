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

import os
import time 
import functools
import numpy as np
from math import cos, pi, floor, sin

import torch
import torch.nn as nn
import torch.nn.functional as F

import re

import torch
import torchaudio.transforms as T

def extract_identifier(filename):
    # extract identifier of the file
    match = re.search(r'_en_(\d+)_SNR_', filename)
    return match.group(1) if match else None

def extract_snr_value(filename):
    # split the string by underscores
    parts = filename.split('_')

    # the SNR value is the last element
    snr_value = parts[-1]
    return snr_value

def flatten(v):
    return [x for y in v for x in y]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def find_final_epoch(path):
    """
    Find latest checkpoint
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    final = -1
    for f in files:
        if not f.startswith('bestmodel_itr_') and f.endswith("final.pkl"):          
            # extract the number part of the filename
            number_part = f.replace('_final.pkl', '')
            try:
                # convert the number to an integer
                number = int(number_part)
                # update final
                if number > final:
                    final = number
            except ValueError:
                continue
    return final

def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True, end="; ")
        
        if keyword is not None:
            keyword_parameters = [p for name, p in net.named_parameters() if p.requires_grad and keyword in name]
            params = sum([np.prod(p.size()) for p in keyword_parameters])
            print("{} Parameters: {:.6f}M".format(
                keyword, params / 1e6), flush=True, end="; ")
        
        print(" ")


####################### lr scheduler: Linear Warmup then Cosine Decay #############################

# Adapted from https://github.com/rosinality/vq-vae-2-pytorch

# Original Copyright 2019 Kim Seonghyeon
#  MIT License (https://opensource.org/licenses/MIT)


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cosine(start, end, proportion):
    cos_val = cos(pi * proportion) + 1
    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, cur_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = cur_iter

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class LinearWarmupCosineDecay:
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        iteration=0,
        divider=25,
        warmup_proportion=0.3,
        phase=('linear', 'cosine'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {'linear': anneal_linear, 'cosine': anneal_cosine}

        cur_iter_phase1 = iteration
        cur_iter_phase2 = max(0, iteration - phase1)
        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, cur_iter_phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, cur_iter_phase2, phase_map[phase[1]]),
        ]

        if iteration < phase1:
            self.phase = 0
        else:
            self.phase = 1

    def step(self):
        lr = self.lr_phase[self.phase].step()

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            self.phase = 0

        return lr


####################### model util #############################

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)


@torch.no_grad()
def sampling(net, noisy_audio):
    """
    Perform denoising (forward) step
    """

    return net(noisy_audio)


@torch.no_grad()
def sampling(net, noisy_audio):
    """
    Perform denoising (forward) step
    """

    return net(noisy_audio)


def loss_fn(net, X):
    """
    Loss function in CleanUNet

    Parameters:
    net: network
    X: training data pair (clean audio, noisy_audio)
    ell_p: \ell_p norm (1 or 2) of the AE loss

    Returns:
    loss: value of objective function
    output_dic: values of each component of loss
    """

    assert type(X) == tuple and len(X) == 2
    
    clean_spec, noisy_spec = X
    output_dic = {}
    loss = 0.0
    
    # AE loss
    denoised_spec = net(noisy_spec) 

    # Mean Squared error
    ae_loss = nn.MSELoss()(denoised_spec, clean_spec)
   
    output_dic["reconstruct"] = ae_loss.data

    loss = ae_loss
    return loss, output_dic

