import os
import sys
import torch
import numpy as np

from .infer_utils import inference, inference_ctc

def calculate_error_rate(phons, gts, hparams):
    seq_gen_vs_gt = []
    nbr_errors = 0
    if hparams['use_beam_search']:
        for i in range(len(gts)):
            if i >= len(phons[0][0]):
                seq_gen_vs_gt.append('-' + ' ' + '-' + ' ' + '-' + ' | ' + gts[i]) 
                nbr_errors+=1
            else:
                seq_gen_vs_gt.append(phons[2][0][i] + ' ' + phons[1][0][i] + ' ' + phons[0][0][i] + ' | ' + gts[i]) 
                if phons[0][0][i]!=gts[i]:
                    nbr_errors+=1
    else:
        for i in range(len(gts)):
            if i >= len(phons):
                seq_gen_vs_gt.append('-' + ' ' + '-' + ' ' + '-' + ' | ' + gts[i]) 
                nbr_errors+=1
            else:
                seq_gen_vs_gt.append(phons[i][0] + ' ' + phons[i][1] + ' ' + phons[i][2] + ' | ' + gts[i])
                if phons[i][2]!=gts[i]:
                    nbr_errors+=1
    # print(seq_gen_vs_gt, nbr_errors)
    return seq_gen_vs_gt, nbr_errors

def save_inference(base_name, name, phons_seq):
    "TO BE DONE"
    return

def write_test_results(hparams, name, base_name, seq_gen_vs_gt):
    nbr_docs = len([entry for entry in os.listdir(hparams['results_dir']) if os.path.isfile(os.path.join(hparams['results_dir'], entry))])
    # with open(f"{hparams['results_dir']}/Test_results_{hparams['model_name']}_{hparams['dataset_name']}_{nbr_docs}_{hparams['lr']}_{hparams['epochs']}_{hparams['n_layers']}", 'w') as file:
    with open(f"{hparams['results_dir']}/{base_name}_" +name , 'w') as file:
        file.write(f"File name :  {name} \n")
        file.write(f"Hparams: {hparams} \n\n\n".replace(',', '\n'))
        for elt in seq_gen_vs_gt:
            file.write(elt + '\n')
        file.close()

def display_test_results(nbr_errors):
    nbr_files = len(nbr_errors)
    total_nbr_errors = 0
    total_nbr_phonemes = 0
    for name, nbr_er in nbr_errors.items():
        print(f'{name} : {nbr_er[0]} errors over {nbr_er[1]} phonemes, a ratio of {nbr_er[0]/nbr_er[1]:.2%}')
        total_nbr_errors+=nbr_er[0]
        total_nbr_phonemes+=nbr_er[1]
    print('Mean absolute number of errors / mean number of phonemes: {:.4} / {:.4}, which represents a ratio of {:.2%}'.format(total_nbr_errors/nbr_files, total_nbr_phonemes/nbr_files, total_nbr_errors/total_nbr_phonemes))
    return total_nbr_errors/total_nbr_phonemes

def evaluation_core(phoneme_list, phon_target, phons, hparams, name, base_name, seq_gen_vs_gt_dic, nbr_errors_dic):
    gts = [phoneme_list[p] for p in phon_target[0]]    # cal_new_lists(phons, gts, rot)
    seq_gen_vs_gt, nbr_errors = calculate_error_rate(phons, gts, hparams)
    seq_gen_vs_gt_dic[name[0]] = seq_gen_vs_gt
    assert len(gts)==len(phon_target[0]), f'gts: {gts}, ground_truth: {phon_target[0]}'
    nbr_errors_dic[name[0]] =  (nbr_errors, len(phon_target[0]))
    save_inference(base_name, name[0], phons)
    write_test_results(hparams, name[0], base_name, seq_gen_vs_gt)
    return nbr_errors_dic

def evaluate_ctc(hparams, test_loader, device, model, phoneme_list, embedded_phonemes, base_name, threshold=0.2):
    loop_number = 0
    nbr_errors_dic = {}
    seq_gen_vs_gt_dic = {}
    for input, onset_target, phon_target, ctc_target, name in test_loader:
        loop_number+=1
        input, onset_target, phon_target, ctc_target = input.to(device), onset_target.to(device), phon_target.to(device), ctc_target.to(device)
        phons = inference_ctc(input, model, hparams, phoneme_list, threshold)
        nbr_errors_dic = evaluation_core(phoneme_list, phon_target, phons, hparams, name, base_name, seq_gen_vs_gt_dic, nbr_errors_dic)
    rate_errors = display_test_results(nbr_errors_dic)
    return seq_gen_vs_gt_dic, nbr_errors_dic, rate_errors

def evaluate(hparams, test_loader, device, model, phoneme_list, embedded_phonemes, base_name):
    loop_number = 0
    nbr_errors_dic = {}
    seq_gen_vs_gt_dic = {}
    for input, ground_truth, name in test_loader:
        print(name)
        loop_number+=1
        input, ground_truth = input.to(device), ground_truth.to(device)
        phons = inference(input, model, hparams, phoneme_list)
        nbr_errors_dic = evaluation_core(phoneme_list, ground_truth, phons, hparams, name, base_name, seq_gen_vs_gt_dic, nbr_errors_dic)
    rate_errors = display_test_results(nbr_errors_dic)

    return seq_gen_vs_gt_dic, nbr_errors_dic, rate_errors
