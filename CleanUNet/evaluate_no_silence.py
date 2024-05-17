import json
import os
from denoise import denoise
from python_eval import evaluate
import csv
import copy

# base configuration path
config_file = 'configs/DNS-large-full.json'
with open(config_file, 'r') as config_file:
    baseline_config = json.load(config_file)

# base directory containing model layer directories
# insert path to base directory of the trained models here
base_directory = ''

# List all directories in the base directory
model_directories = [dir for dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, dir))]

for model_dir in model_directories:
    model_path = os.path.join(base_directory, model_dir)
    
    checkpoint_directory = os.path.join(model_path, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_directory, '152396_final.pkl')
    
    # get number of layers from name of the directory
    num_layers = int(model_dir.split('_')[2])
    
    # change config to number of required layers
    config = copy.deepcopy(baseline_config)
    config['network_config']['encoder_n_layers'] = num_layers
    network_config = config['network_config']
    trainset_config = config['trainset_config']

    # denoise the audios
    denoise(ckpt_iter='final', subset='testing', ckpt_directory=checkpoint_directory, dump=True, 
            train_config=config['train_config'], trainset_config=trainset_config, 
            network_config=network_config)

    # evaluate the model
    # insert path to testdataset here
    testset_path = ''
    enhanced_path = os.path.join(model_path, 'enhanced_speech_no_silence_152396')
    results = evaluate(testset_path, enhanced_path, target='enhanced')

    # get csv path
    csv_file_path = os.path.join(model_path, 'results_no_silence.csv')
    
    # write results to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'STOI', 'MSE'])
        writer.writerow([model_dir, results['stoi'], results['mse']])

    print(f"Results saved to {csv_file_path}")
