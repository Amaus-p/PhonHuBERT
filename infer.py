import torch
import os
import sys

sys.path.insert(0, os.path.abspath('../'))
from utils.eval_utils import evaluate, evaluate_ctc
from utils.model_loading import load_model
from utils.constants import HParams
from utils.data_loader import MusicDataLoader
import time


def load_all_checkpoints():
    # Replace 'your_folder_path' with the path to the folder you want to search in
    folder_path = './new_checkpoints/'
    fn_checkpoints = []

    # Use os.walk() to recursively find all subfolders
    i=0
    directories = []
    for root, dirs, files in os.walk(folder_path):
        directories = dirs
        break
    for i in range(len(directories)):
        for root, dirs, files in os.walk(folder_path+directories[i]):
            for file in files:
                fn_checkpoints.append(os.path.join(root, file))

    print(fn_checkpoints)
    return fn_checkpoints

def create_hparams_from_old_checkpoint():
    hparams = HParams(   
        model_name='LSTM',     
        dataset_name = 'opencpop',    
        segmented = False,
        seq_length = None   
    )
    hparams['epochs'] = 1000
    return hparams

def load_model_checkpoint(checkpoint, device):
    print(f"Loading checkpoint: {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    hparams = checkpoint['hparams']
    if not 'use_ctc_onset' in hparams:
        hparams['use_ctc_onset'] = False
    if not 'load_audio_plus_hubert' in hparams:
        hparams['load_audio_plus_hubert'] = False
    # print(hparams)
    model, _,  _ = load_model(device, hparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Checkpoint EPOCH', checkpoint['epoch'])
    model.eval()
    return model, hparams


def get_device_from_checkpoint(checkpoint_path):
    # Load a part of the state_dict to inspect
    state_dict_sample = torch.load(checkpoint_path, map_location='cpu')
    
    # Get a tensor from the loaded state_dict
    tensor_name, tensor = next(iter(state_dict_sample.items()))
    
    return tensor.device


if __name__ == "__main__":
    device = torch.device('cpu')

    checkpoint_names = []
    #PUT THE CHECKPOINT NAMES HERE
    checkpoint_name = "2024-01-30_10-42-36-880156/LSTM_0.0001_1000_2_1706582556.pth"
    checkpoint_names.append(checkpoint_name)

    #To load all checkpoints in the folder
    # checkpoint_names = load_all_checkpoints()

    # device = get_device_from_checkpoint(checkpoint)
    # print(f"The device of the checkpoint is: {device}")

    best_setup_cp = {}
    thresholds=[0.02]
    for checkpoint_name in checkpoint_names:
        print(checkpoint_name)
        min_error_rate = 100

        checkpoint = f"./new_checkpoints/{checkpoint_name}"
        # checkpoint = f"{checkpoint_name}"

        model, hparams = load_model_checkpoint(checkpoint, device)
        hparams['max_seq_length']=16094
        hparams['max_length']=321.88
        hparams['use_beam_search']=False
        if not 'use_same_dim_pred_target' in hparams:
            hparams['use_same_dim_pred_target'] = False
        if not 'new_new_model' in hparams:
            hparams['new_new_model'] = False
        if not 'softmax' in hparams:
            hparams['softmax'] = False
        print(hparams)
        # print(model)
        test_dataset = MusicDataLoader(hparams, is_test=True)
        test_loader, phon_list, embedded_phonemes = test_dataset.main_create_phon_set()
        results_dir = hparams['results_dir']+checkpoint_name.split('/')[0].replace('.pt', '')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        base_name= f"{checkpoint_name.split('/')[0].replace('.pt', '')}/{hparams['model_name'] }_{hparams['dataset_name']}_{hparams['lr']}_{hparams['epochs']}_{hparams['n_layers']}_{str(time.time()).split('.')[0]}_evaluation"
       
        if hparams['use_ctc_onset']:
            for t in thresholds:
                print('Threshold', t)
                metrix = evaluate_ctc(hparams, test_loader, device, model, phon_list, embedded_phonemes, base_name, threshold=t)
                seq_gen_vs_gt_dic, nbr_errors_dic, rate_errors = metrix
                if rate_errors < min_error_rate:
                    min_error_rate = rate_errors
                    best_threshold = t
                    best_metrix = metrix 
            best_setup_cp[checkpoint_name] = [best_threshold, min_error_rate, best_metrix]
            for k, v in best_setup_cp.items():
                print(k, v[0], v[1])   
            with open(f"{results_dir}/finals.txt", 'w') as f:
                f.write(str(rate_errors))
                f.write('\n')
                f.write(str(hparams))
                f.write('\n')
                f.write(str(checkpoint))
                f.write('\n')
                f.write(str(best_setup_cp))
                f.write('\n')
                f.write(str(thresholds))
            f.close

        else:
            metrix = evaluate(hparams, test_loader, device, model, phon_list, embedded_phonemes, base_name)
            seq_gen_vs_gt_dic, nbr_errors_dic, rate_errors = metrix

            with open(f"{results_dir}/finals.txt", 'w') as f:
                f.write(str(rate_errors))
                f.write('\n')
                f.write(str(hparams))
                f.write('\n')
                f.write(str(checkpoint))
            f.close





