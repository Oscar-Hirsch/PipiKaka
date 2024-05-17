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
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import torchaudio.transforms as transformsaudio

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms
import torchaudio
import torch.nn.functional as F

from util import extract_identifier, extract_snr_value



class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean waveform, noisy waveform, file_id)
    """
    
    def __init__(self, root='./', subset='training', crop_length_sec=0):
        # get the audio files and sort them so that we have two lists (noisy and clean) with the the clean audio files and the corresponding noisy audio files at the same location
        super(CleanNoisyPairDataset).__init__()

        assert subset is None or subset in ["training", "testing"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset
    
        if subset == "training":
            noisy_folder = os.path.join(root, 'Datasets/training_set/noisy_data')
            clean_folder = os.path.join(root, 'Datasets/training_set/clean_data')
        
        elif subset == "testing":
            noisy_folder = os.path.join(root, 'Datasets/test_set/noisy_data')
            clean_folder = os.path.join(root, 'Datasets/test_set/clean_data')

        else:
            raise NotImplementedError
        
        noisy_files = os.listdir(noisy_folder)
        self.files = []

        for nf in noisy_files:
            # get file name using extract_identifier and extract_snr_value functions
            identifier = extract_identifier(nf)
            snr_value = extract_snr_value(nf)
            clean_file = f"common_voice_en_{identifier}_SNR_{snr_value}"
            clean_file_path = os.path.join(clean_folder, clean_file)
            noisy_file_path = os.path.join(noisy_folder, nf)

            if os.path.exists(clean_file_path):
                self.files.append((clean_file_path, noisy_file_path))
            else:
                print(f"Warning: Clean file {clean_file_path} not found for {noisy_file_path}")
        


    def __getitem__(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0])
        noisy_audio, sample_rate = torchaudio.load(fileid[1])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * sample_rate)   
        
        # pad to 5 seconds
        if self.subset != 'testing' and crop_length > 0:
            if len(clean_audio) < crop_length:
                pad_size = crop_length - len(clean_audio)
                clean_audio = F.pad(clean_audio, (0, pad_size), 'constant', 0)
                noisy_audio = F.pad(noisy_audio, (0, pad_size), 'constant', 0)

            start = 0
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]
        
        # compute magnitude spectrogram
        transform = transformsaudio.Spectrogram(n_fft=512, hop_length=256, power=1)
        clean_spectrogram = transform(clean_audio)
        noisy_spectrogram = transform(noisy_audio)
        

        clean_spectrogram, noisy_spectrogram = clean_spectrogram.unsqueeze(0), noisy_spectrogram.unsqueeze(0)
        

        return (clean_spectrogram, noisy_spectrogram, fileid)

    def __len__(self):
        return len(self.files)

def load_CleanNoisyPairDataset(root, subset, crop_length_sec, batch_size, sample_rate, num_gpus=1):
    # Create dataloaders to iterate over the audio files, with possible use of multiple GPUs for faster processing
    """
    Get dataloader with distributed sampling
    """
    dataset = CleanNoisyPairDataset(root=root, subset=subset, crop_length_sec=crop_length_sec)                                                       
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        
    return dataloader


if __name__ == '__main__':
    # prints information about the first batch of audio files when preparing the dataloaders (dataloader for iterating over the audio files)
    import json
    with open('configs/DNS-large-full.json') as f:
        data = f.read()
    config = json.loads(data)
    trainset_config = config["trainset_config"]

    trainloader = load_CleanNoisyPairDataset(**trainset_config, subset='training', batch_size=2, num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config, subset='testing', batch_size=2, num_gpus=1)
    print(len(trainloader), len(testloader))

    for clean_spec, noisy_spec, fileid in trainloader: 
        clean_spec = clean_spec.cuda()
        noisy_spec = noisy_spec.cuda()
        print(clean_spec.shape, noisy_spec.shape, fileid)
        break
