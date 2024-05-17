import json
import copy  
import train_model
from train_model import train  
from denoise import denoise
from python_eval import evaluate
import csv
import os

# load config
config_file_path = 'configs/DNS-large-full.json' 
with open(config_file_path, 'r') as config_file:
    base_config = json.load(config_file)

encoding_layers_configurations = [5, 6, 7, 8]
num_runs_per_configuration = 5

# loop over each encoding layer option
for num_layers in encoding_layers_configurations:
    # loop for 5 runs
    for run in range(num_runs_per_configuration):
        config = copy.deepcopy(base_config)

        # ppdate the network configs
        config['network_config']['encoder_n_layers'] = num_layers

        # extract updated configs
        network_config = config['network_config']
        train_config = config['train_config']
        trainset_config = config['trainset_config']
        dist_config = config['dist_config']

        train_model.network_config = network_config
        train_model.trainset_config = trainset_config
        train_model.dist_config = dist_config

        # train the model
        ckpt_directory = train(num_gpus=1, rank=0, group_name='', 
              exp_path=train_config['exp_path'], 
              log=train_config['log'], 
              optimization=train_config['optimization'], 
              loss_config=train_config['loss_config'])

        # denoise the model
        denoise(ckpt_iter='final', subset='testing', ckpt_directory=ckpt_directory, dump=True, train_config=train_config, trainset_config = trainset_config, network_config=network_config)

        # evaluate the model
        # insert testset path here
        testset_path = ''
        helper_model_path, checkpoint_name = os.path.split(ckpt_directory)
        enhanced_path = os.path.join(helper_model_path, f"enhanced_speech")
        target = 'enhanced'  
        results = evaluate(testset_path, enhanced_path, target)

        # get checkpoint path name
        model_dir, enhanced_name = os.path.split(enhanced_path)

        # get csv path
        csv_file_path = os.path.join(model_dir, f"results_{enhanced_name}.csv")

        # write results to the CSV file
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'STOI', 'MSE'])
            writer.writerow([enhanced_name, results['stoi'], results['mse']])

        print(f"Results saved to {csv_file_path}")


