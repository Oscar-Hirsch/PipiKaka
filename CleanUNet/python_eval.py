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
import sys
from collections import defaultdict
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
import csv

import numpy as np
import torch.nn as nn
import torch
from scipy.io import wavfile 
import torchaudio
from pystoi.pystoi.stoi import stoi

def evaluate(testset_path, enhanced_path, target):

    result_stoi = 0
    result_mse = 0
    result = defaultdict(int)

    # list all files in the clean directory 
    clean_files = os.listdir(os.path.join(testset_path, "clean_data_spec"))       
    print(f'Testset: {testset_path}')
    print(f'Enhanced: {enhanced_path}')

    for clean_file in tqdm(clean_files):
        try:
            # Load clean spectrogram 
            clean_spectrogram = np.load(os.path.join(testset_path, "clean_data_spec", clean_file))
            # file ID
            file_id = clean_file.split('_')[3]  
            # SNR value
            snr_value = clean_file.split('_')[5].split('.')[0]  

            if target == 'noisy':
                noisy_file = f"noisy_common_voice_en_{file_id}_SNR_{snr_value}.npy"  
                target_spectrogram = np.load(os.path.join(testset_path, "noisy_data", noisy_file))
            else:
                enhanced_file = f"enhanced_voice_en_{file_id}_SNR_{snr_value}.npy"  
                target_spectrogram = np.load(os.path.join(enhanced_path, enhanced_file))
        except:
            continue

        result_stoi += stoi(clean_spectrogram, target_spectrogram)

        clean_spectrogram = torch.from_numpy(clean_spectrogram).float()
        target_tensor = torch.from_numpy(target_spectrogram).float()

        result_mse += nn.MSELoss()(target_tensor, clean_spectrogram).item()
    
    result['stoi'] += result_stoi/len(clean_files)
    result['mse'] += result_mse/len(clean_files) 

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='dns', help='dataset')
    parser.add_argument('-e', '--enhanced_path', type=str, help='enhanced audio path')
    parser.add_argument('-t', '--testset_path', type=str, help='testset path')
    args = parser.parse_args()

    enhanced_path = args.enhanced_path
    testset_path = args.testset_path
    target = 'enhanced'

    if args.dataset == 'dns':
        result = evaluate(testset_path, enhanced_path, target)
        
    # logging
    for key in result:
        if key != 'count':
            print('{} = {:.9f}'.format(key, result[key]), end=", ")


    # get checkpoint path name
    model_dir, enhanced_name = os.path.split(enhanced_path) 
    
    # safe csv  
    csv_file_path = os.path.join(model_dir, f"results_{enhanced_name}.csv")

    
    # write results to csv
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'STOI', 'MSE'])  
        writer.writerow([model_dir, result['stoi'], result['mse']]) 
