import torch
import numpy as np

def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists

def create_sequence_ground_truth(embedded_groud_truth):
    gt_seq = []
    current = ''
    for phon in embedded_groud_truth:
        if phon!=current and phon!='SP' and phon!='AP':
            current = phon
            gt_seq.append(current)

    with open('./gt_seq','w') as f:
        for e in gt_seq:
            f.write(str(e) + '\n')
    return gt_seq

def output_to_phon(out, phoneme_list):
    phons = []
    for vec in out[0]:
        arg_max_sorted = torch.argsort(vec, descending=False)
        phons.append([phoneme_list[i] for i in arg_max_sorted[len(arg_max_sorted)-3:]])
    return phons

def inference(example, model, hparams, phoneme_list):
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)
    model.eval()
    with torch.no_grad():
        if hparams['hidden']:
            test_h = model.init_hidden(1)
            out, test_h = model(example, test_h)
        else:
            out = model(example)
    return output_to_phon(out, phoneme_list)

def onset_ctc_cal(onset_out, threshold=0.2, window = 2):
    onset_pred = onset_out[0,:,1].cpu().detach().numpy()
    onset_lst = []
    i=0
    onset = 0
    while i < len(onset_pred):
        if onset_pred[i]>threshold:
            onset = np.argmax(onset_pred[max(i-window, 0): min(i+window, len(onset_pred)-1)])+max(i-window, 0)
            onset_lst.append(onset)
            i = i+window*2+1
        else:
            i+=1
    return onset_lst

def phon_ctc_cal(onset_lst, phon_out, phon_list_embedded):
    phon_pred = phon_out[0].cpu().detach().numpy()
    phon_lst = []
    for i in range(len(onset_lst)):
        start = onset_lst[i]
        stop = onset_lst[i+1] if i<len(onset_lst)-1 else len(phon_pred)
        temp = [np.argmax(phon_pred[j, :]) for j in range(start, stop)]
        temp = np.bincount(temp)
        for j in range(start, stop):    
            phon_lst.append([phon_list_embedded[np.argmax(temp)] for i in range(3)])
    return phon_lst

def inference_ctc(input, model, hparams, phon_list, threshold=0.2):
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)
    model.eval()
    with torch.no_grad():
        if hparams['hidden']:
            [onset_hc, phon_hc] = model.init_hidden(1)
            onset_out, phon_out, onset_hc, phon_hc = model.infer(input, [onset_hc, phon_hc])
            onset_lst = onset_ctc_cal(onset_out, threshold)
            phon_lst_seq = phon_ctc_cal(onset_lst, phon_out, phon_list)
    return phon_lst_seq
