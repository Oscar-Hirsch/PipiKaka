# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

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
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from datetime import datetime

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from dataset import load_CleanNoisyPairDataset

from util import rescale, print_size
from util import LinearWarmupCosineDecay, loss_fn

from network import CleanUNet

def train(num_gpus, rank, group_name, 
          exp_path, log, optimization, loss_config):

    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)
    

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    encoding_layers = network_config['encoder_n_layers']
    # Get shared ckpt_directory ready
    current_time = datetime.now().strftime("%Y-%m-%d_%H")
    # insert base checkpoint directory here
    ckpt_base_directory = ''
    ckpt_directory = os.path.join(ckpt_base_directory, f"Model_Layers_{encoding_layers}_Time_{current_time}", "checkpoints")
    print(ckpt_directory)

    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    trainloader = load_CleanNoisyPairDataset(**trainset_config, 
                            subset='training',
                            batch_size=optimization["batch_size_per_gpu"], 
                            num_gpus=num_gpus)
    print(f'Trainingdata loaded: {len(trainloader)}')

    for batch in trainloader:
        # extract a batch of spectrograms
        clean_spectrograms, noisy_spectrograms, fileids = batch

        # print the shape of spec
        height = clean_spectrograms.shape[2]
        width = clean_spectrograms.shape[3]
        break
    print(f'Before encoder (height/width): ({height},{width})')
    
    # model
    net = CleanUNet(**network_config).cuda()
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

    # load checkpoint
    time0 = time.time()
    ckpt_iter = -1
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # training
    n_iter = ckpt_iter + 1

    # define learning rate scheduler and stft-loss
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=optimization["learning_rate"],
                    n_iter=optimization["n_iters"],
                    iteration=n_iter,
                    divider=25,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )
    
    checkpoint_saving_time = 0
    pre_checkpoint = 0
    post_checkpoint = 0
    epoch = 0
    while n_iter < optimization["n_iters"]+1:
        epoch += 1

        
        # for each epoch
        for clean_audio, noisy_audio, _ in trainloader: 
            print(f'Current Epoch: {epoch}, Iterations: {n_iter}')
            
            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()

            # back-propagation
            optimizer.zero_grad()
            X = (clean_audio, noisy_audio)
            loss, loss_dic = loss_fn(net, X)
            reduced_loss = loss.item()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            scheduler.step()
            optimizer.step()


            n_iter += 1
	    

    # Save last checkpoint after the last iteration
    if rank == 0:
        pre_checkpoint = time.time()
        final_checkpoint_name = '{}_final.pkl'.format(n_iter-1)
        torch.save({'iter': n_iter-1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_time_seconds': int(time.time()-time0), 
                    'checkpoint_time_seconds': checkpoint_saving_time,
                    'Reduced loss': reduced_loss},
                    os.path.join(ckpt_directory, final_checkpoint_name))
        post_checkpoint = time.time()
        checkpoint_saving_time = checkpoint_saving_time + (post_checkpoint-pre_checkpoint)
        print(f'Model at iteration {n_iter-1} is saved as final checkpoint. Reduced Loss: {reduced_loss}')


    return ckpt_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/DNS-large-full.json', 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
         data = f.read()
    config = json.loads(data)
    train_config            = config["train_config"]        # training parameters
    global dist_config
    dist_config             = config["dist_config"]         # to initialize distributed training
    global network_config
    network_config          = config["network_config"]      # to define network
    global trainset_config
    trainset_config         = config["trainset_config"]     # to load trainset


    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # execute train function
    train(num_gpus, args.rank, args.group_name, **train_config)
