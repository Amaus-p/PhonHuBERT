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
    folder_path = '/home/yunkaiji/data2/PhonHuBERT_no_git/new_checkpoints/'
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
    # checkpoint_name = "2024-01-27_20-13-28-881409/LSTM_0.001_1000_2_1706357608.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name=  "2024-01-28_00-43-07-186450/LSTM_0.001_1000_2_1706373787.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_names.append("2024-01-28_01-09-53-868934/LSTM_0.0001_1000_2_1706375393.pth")
    # checkpoint_names.append("2024-01-28_11-45-30-145541/LSTM_0.01_1000_2_1706413530.pth")
    # checkpoint_name= "2024-01-28_13-33-39-220394/LSTM_0.001_1000_6_1706420019.pth"
    # checkpoint_name= "2024-01-28_12-16-56-885467/LSTM_0.001_1000_6_1706415416.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name="2024-01-28_11-57-44-119258/LSTM_0.001_1000_6_1706414264.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-28_20-07-38-780307/LSTM_0.001_1000_2_1706443658.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_09-35-29-288700/LSTM_0.001_1000_2_1706492129.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_09-36-28-470572/LSTM_0.001_1000_2_1706492188.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_12-05-29-127067/LSTM_0.001_1000_2_1706501129.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_09-38-08-908911/LSTM_0.001_1000_2_1706492288.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_11-59-40-684899/LSTM_0.001_1000_2_1706500780.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-28_21-19-15-858159/LSTM_0.001_1000_2_1706447955.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2023-10-21_15-10-25-272262/LSTM_0.001_3000_2_1697872225.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-28_12-58-36-436330/LSTM_0.001_1000_6_1706417916.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-28_21-16-50-967936/LSTM_0.001_1000_6_1706447810.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-10-11-919924/LSTM_0.001_1000_4_1706533811.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-30-47-810930/LSTM_0.001_1000_2_1706535047.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-12-04-849658/LSTM_0.001_1000_2_1706533924.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-12-24-530384/LSTM_0.001_1000_2_1706533944.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-14-20-078512/LSTM_0.001_1000_2_1706534060.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-14-34-583044/LSTM_0.001_1000_2_1706534074.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-01-13-926594/LSTM_0.001_1000_2_1706533273.pth"
    # checkpoint_names.append(checkpoint_name)
    checkpoint_name = "2024-01-30_10-42-36-880156/LSTM_0.0001_1000_2_1706582556.pth"
    checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-29_21-15-28-458076/LSTM_0.001_1000_2_1706534128.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_10-40-54-008943/LSTM_0.001_1000_2_1706582454.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_11-06-09-066346/LSTM_0.001_1000_2_1706583969.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_11-06-34-450949/LSTM_0.001_1000_2_1706583994.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_11-07-05-271651/LSTM_0.001_1000_2_1706584025.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_15-50-18-731380/LSTM_0.001_1000_2_1706601018.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_15-40-36-704344/LSTM_0.001_1000_4_1706600436.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_15-12-29-218798/LSTM_0.001_1000_6_1706598749.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_name = "2024-01-30_15-38-39-980693/LSTM_0.001_1000_6_1706600319.pth"
    # checkpoint_names.append(checkpoint_name)
    # checkpoint_names.append("2024-01-30_14-13-01-283710/LSTM_0.001_1000_4_1706595181.pth")
    # checkpoint_names.append("2024-01-30_15-03-33-827238/LSTM_0.001_1000_6_1706598213.pth")
    # checkpoint_names.append("2024-01-29_21-30-47-810930/LSTM_0.001_1000_2_1706535047.pth")
    # checkpoint_names.append("2024-01-30_23-04-36-184722/LSTM_0.001_1000_2_1706627076.pth")

    # print(checkpoint_names)

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





