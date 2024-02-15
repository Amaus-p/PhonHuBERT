from typing import Dict

def HParams(
    model_name='LSTM',
    dataset_name = 'opencpop',    
    segmented = False,
    seq_length = None
) -> Dict[str, any]:
    model_names = ['LSTM', 'double_LSTM']
    dataset_names = ['opencpop']

    use_ctc_onset = False
    new_new_model = True

    #Select the decoder model
    use_hubert = True
    use_mel = False

    #Select the loss function
    use_mse_loss= True  #False
    use_focal_loss = False #True
    use_cross_entropy = False  #True

    #Softmax is included in the cross entropy loss
    softmax = True

    assert use_mse_loss and softmax or not use_mse_loss, 'Must use softmax with mse loss'    

    assert not (use_mse_loss and use_focal_loss), 'Cannot use both mse loss and focal loss'
    assert not (use_mse_loss and use_cross_entropy), 'Cannot use both mse loss and cross entropy'
    assert not (use_focal_loss and use_cross_entropy), 'Cannot use both focal loss and cross entropy'

    use_same_dim_pred_target = use_focal_loss or use_mse_loss
    # sample_rate = 16000
    sample_rate = 44100
    hop_length = 512

    assert use_hubert or use_mel, 'Must use hubert or mel'
    assert not (use_hubert and use_mel), 'Cannot use both hubert and mel'

    assert model_name in model_names, f'Invalid model name: {model_name}'
    assert dataset_name in dataset_names, f'Invalid dataset name: {dataset_name}'

    hidden = model_name in ['LSTM', 'RNN', 'double_LSTM']

    if use_hubert:
        frames_per_sec = 50
        input_dim = 256
    elif use_mel:
        input_dim = 229
        frames_per_sec = sample_rate/hop_length

    if dataset_name == 'opencpop':            
        data_root_dir="./data/opencpop/"
        # data_root_dir="./data/opencpop_hubert/"
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
        'use_hubert': use_hubert,
        'use_mel': use_mel,

        #data parameters
        'data_root_dir': data_root_dir,
        'results_dir': './results/',
        'dataset_name': dataset_name,
        'segmented': False,
        'padding': True,        
        'seed': 10,
        'seq_length': seq_length,
        'load_audio_plus_hubert': False,
        'frames_per_sec': frames_per_sec,

        #checkpoints parameters
        'checkpoint_dir': './new_checkpoints/',

        #training parameters
        'epochs': 1000,
        'print_every': 20,
        'clip': 5,
        'lr': 1e-3,

        #spectrogram parameters
        'sample_rate': sample_rate,
        'features': 'spectrogram',
        'spec_type': 'mel',
        'spec_mel_htk': False,
        'spec_diff': False,
        'spec_log_scaling': True,
        'spec_hop_length': hop_length,
        'spec_n_bins_mel': 229,
        'spec_n_fft': 2048,
        'spec_fmin': 55.0,

        #lstm parameters
        'input_dim': input_dim,
        'hidden_dim': input_dim//2,
        'output_dim': 60,
        'batch_size': batch_size,
        'n_layers': 2,
        'bidirectional': True,
        'drop_prob': 0.5,
        'batch_first': True,
        'note': 'No note',
        'hidden': hidden,
        'softmax': softmax,

        #local attention parameters
        'max_seq_length': 21634, #depends on the data 
        # 'max_seq_length': 16094, #for the test data
        'max_length': 432.68,
        'coef_for_mel': 86.130627715632800221872977720255,
        'window_size': 373,
        'causal': False,
        'look_backward': 1,
        'look_forward': 1,
        'exact_windowsize': False,

        #loss parameters
        'l_ctc': 0.5,
        'l_ce': 0.5,
        'ep_start_ctc': 5,
        'use_ctc_onset': use_ctc_onset,
        'use_focal_loss': use_focal_loss,
        'use_cross_entropy': use_cross_entropy,
        'use_mse_loss': use_mse_loss,
        'use_same_dim_pred_target': use_same_dim_pred_target,
        'scheduler_patience': 50,
        "scheduler": True,
        #inference params
        'use_beam_search': False,

    }