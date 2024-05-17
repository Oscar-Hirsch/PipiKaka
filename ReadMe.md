## Descrition

The network presented in this code was taken from 10.1109/ICASSP43922.2022.9746169 and adapted to work with 2D magnitude spectrograms.
It it based on a UNet architecture with a transformer bottleneck. 

## how to run train_runs.py
This script trains and evaluates the models.
To run the scrip one has insert the corresponding testset path in the line testset_path = ''.
Further the correct root path in the DNS-large-full.json configuration file has to be set which aims to the directory where the Datasets for training and testing are stored. 
Also it has to be assured that the correct paths for the trainingset and testset are set in dataset.py. 
The specific lines are:
* noisy_folder = os.path.join(root, 'Datasets/test_set/noisy_data')
* clean_folder = os.path.join(root, 'Datasets/test_set/clean_data')

## How to run evaluate_no_silence
This only script evaluates the models. We used it to evaluate the models on a different testset, namely one without silence. 

To run the scrip one has insert the corresponding testset path in the line testset_path = ''.
Further the correct root path in the DNS-large-full.json configuration file has to be set which aims to the directory where the Datasets for training and testing are stored. 
Also it has to be assured that the correct paths for the trainingset and testset are set in dataset.py. 
The specific lines are:
* noisy_folder = os.path.join(root, 'Datasets/test_set/noisy_data')
* clean_folder = os.path.join(root, 'Datasets/test_set/clean_data')

