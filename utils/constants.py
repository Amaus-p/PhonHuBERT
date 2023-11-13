from typing import Dict

def HParams(
    model_name='LSTM',     
    dataset_name = 'opencpop',    
    segmented = False,
    seq_length = None
) -> Dict[str, any]:
    model_names = ['LSTM']
    dataset_names = ['opencpop']
    use_ctc_onset = True
    new_new_model = True

    assert model_name in model_names, f'Invalid model name: {model_name}'
    assert dataset_name in dataset_names, f'Invalid dataset name: {dataset_name}'

    hidden = model_name in ['LSTM', 'RNN']

    if dataset_name == 'opencpop':            
        data_root_dir="./data/opencpop_hubert/"
        if segmented:
            batch_size=10
        else:
            batch_size=1

    return {        
        #which model
        'model_names': model_names,
        'model_name': model_name,
        'use_lstm': True,
        'new_new_model': new_new_model,

        #data parameters
        'data_root_dir': data_root_dir,
        'results_dir': './results/',
        'dataset_name': dataset_name,
        'segmented': False,
        'padding': True,        
        'seed': 10,
        'seq_length': seq_length,
        'load_audio_plus_hubert': False,

        #checkpoints parameters
        'checkpoint_dir': './new_checkpoints/',

        #training parameters
        'epochs': 2000,
        'print_every': 20,
        'clip': 5,
        'lr': 1e-3,

        #lstm parameters
        'input_dim': 256,
        'hidden_dim': 256//2,
        'output_dim': 60,
        'batch_size': batch_size,
        'n_layers': 2,
        'bidirectional': True,
        'drop_prob': 0.5,
        'batch_first': True,
        'note': 'No note',
        'hidden': hidden,

        #loss parameters
        'l_ctc': 0.5,
        'l_ce': 0.5,
        'ep_start_ctc': 5,
        'use_ctc_onset': use_ctc_onset,
    }