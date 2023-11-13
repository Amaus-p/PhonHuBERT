#libraries
import os
import torch

import sys

sys.path.insert(0, os.path.abspath('../'))
from utils.train_utils import train_model, train_model_ctc
from utils.eval_utils import evaluate, write_test_results
from utils.model_loading import load_model
from utils.constants import HParams
from utils.data_loader import MusicDataLoader

import argparse
import time

model_names = ['LSTM', 'RNN', 'Local_Attention', 'Multihead_Attention']

def main_phoneme_seq_cal_train(hparams, device = 'cuda'):
    print(f"MODEL : {hparams['model_name']}")
    print('Collecting the data...')
    dataset = MusicDataLoader(hparams)
    print('Name of the dataset :', dataset.dataset_name)
    if hparams['seq_length']:
        train_loader, val_loader, phoneme_list, _, _ = dataset.main_create_phon_set()
    else:
        train_loader, val_loader, phoneme_list, _, seq_length = dataset.main_create_phon_set()
    
    print(f'Seq length: {seq_length}')
    out_dim = len(phoneme_list)
    hparams['out_dim'] = out_dim
    hparams['seq_length'] = seq_length
    print('Loading the model...')
    model, criterion, optimizer = load_model(
        device,
        hparams)
    print('Training the model...')
    base_name= f"{hparams['model_name'] }_{hparams['lr']}_{hparams['epochs']}_{hparams['n_layers']}_{str(time.time()).split('.')[0]}"
    if hparams['use_ctc_onset']:
        model = train_model_ctc(hparams, device, train_loader, val_loader, model, optimizer, base_name)
    else:
        model = train_model(hparams, device, train_loader, val_loader, model, criterion, optimizer, base_name)
    print("Training ended")
    return

if __name__ == '__main__': 
    print("We are in the main")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f"We use cuda:{DEVICE}")

    hparams = HParams()
    print(hparams)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    main_phoneme_seq_cal_train(hparams, device=device)
