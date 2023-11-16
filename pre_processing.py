import os
import glob

from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocessing.hubertinfer import HubertEncoder
from utils.infer_utils import get_end_file
import librosa

data_dir = "data/opencpop/"

def find_max_wav_len(wav_paths):
    print("Finding max wav length")
    max_length = 0
    step = 0
    max_step = len(wav_paths)
    for wav_path in wav_paths:
        print(wav_path)
        print(f'\r step/number of step : {step}/{max_step}', end='')
        wav, sr = librosa.load(wav_path, sr=None)
        wav16 = librosa.resample(wav, sr, 16000) if sr != 16000 else wav
        length = len(wav16)
        if length>max_length:
            max_length=length
        step+=1
    return max_length

def convert_wav(path=data_dir, max_wav_length=0):
    """
    Taken from the file pre-hubert.py
    I don't use the function itself to be able to manipulate it

    I MODIFY EVERY NPY FILE SO THAT they are all of the same length by adding silences at the end of the wavs files
    """
    print(path)
    hubert_model = HubertEncoder(hubert_mode='soft_hubert')
    wav_paths = get_end_file(path, ".wav")
    print(wav_paths)
    if not max_wav_length:
        max_len = find_max_wav_len(wav_paths) + 100
    else:
        max_len = max_wav_length
    print('Maximum wav length', max_len)
    with tqdm(total=len(wav_paths)) as p_bar:
        p_bar.set_description('Processing HuBERT hidden space vectors')
        for wav_path in wav_paths:
            npy_path = Path(wav_path).with_suffix(".npy")
            if not os.path.exists(npy_path):
                np.save(str(npy_path), hubert_model.encode(wav_path, max_len))
            p_bar.update(1)

def train_test_split(path):
    data_wavs = path + "wavs"
    data_textgrids = path + "textgrids"
    data_midis = path + "midis"
    #create the train and test folders and put in them the wavs, textgrids and midis files, with files 2009, 2016, 2047, 2054, 2087 in the test folder
    #create the train and test folders
    train_dir = path + "train"
    test_dir = path + "test"

    train_wavs = train_dir + "/wavs"
    train_textgrids = train_dir + "/textgrids"
    train_midis = train_dir + "/midis"
    
    test_wavs = test_dir + "/wavs"
    test_textgrids = test_dir + "/textgrids"
    test_midis = test_dir + "/midis"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        os.mkdir(train_wavs)
        os.mkdir(train_textgrids)
        os.mkdir(train_midis)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        os.mkdir(test_wavs)
        os.mkdir(test_textgrids)
        os.mkdir(test_midis)

    wav_paths = get_end_file(data_wavs, ".wav")
    textgrid_paths = get_end_file(data_textgrids, ".TextGrid")
    midi_paths = get_end_file(data_midis, ".midi")
 
    for wav_path in wav_paths:
        if "2009" in wav_path or "2016" in wav_path or "2047" in wav_path or "2054" in wav_path or "2087" in wav_path:
            #check whether the file is already in the test folder
            if not os.path.exists(test_wavs + "/" + wav_path.split("/")[-1]):
                os.system("cp " + wav_path + " " + test_wavs)
        else:
            #check whether the file is already in the train folder
            if not os.path.exists(train_wavs + "/" + wav_path.split("/")[-1]):
                os.system("cp " + wav_path + " " + train_wavs)
    for textgrid_path in textgrid_paths:
        if "2009" in textgrid_path or "2016" in textgrid_path or "2047" in textgrid_path or "2054" in textgrid_path or "2087" in textgrid_path:
            #check whether the file is already in the test folder
            if not os.path.exists(test_textgrids + "/" + textgrid_path.split("/")[-1]):
                os.system("cp " + textgrid_path + " " + test_textgrids)
        else:
            #check whether the file is already in the train folder
            if not os.path.exists(train_textgrids + "/" + textgrid_path.split("/")[-1]):
                os.system("cp " + textgrid_path + " " + train_textgrids)
    for midi_path in midi_paths:
        if "2009" in midi_path or "2016" in midi_path or "2047" in midi_path or "2054" in midi_path or "2087" in midi_path:
            #check whether the file is already in the test folder
            if not os.path.exists(test_midis + "/" + midi_path.split("/")[-1]):
                os.system("cp " + midi_path + " " + test_midis)
        else:
            #check whether the file is already in the train folder
            if not os.path.exists(train_midis + "/" + midi_path.split("/")[-1]):
                os.system("cp " + midi_path + " " + train_midis)
    print("train test split done")

def remove_hubert_files(data_dir):
    print("Removing Hubert vectors")
    current_directory = os.getcwd()
    print(current_directory)
    pattern = os.path.join(data_dir + 'PLEASE REMOVE THIS PART', 'wavs', '*.npy')
    print(pattern)
    files = glob.glob(pattern)
    print(files)
    #remove files from pattern
    for f in files:
        print(f)
        os.remove(f)

if __name__ == '__main__':
    #create the train test split
    train_test_split(data_dir)
    #convert the wavs files into hidden-hubert space vectors
    pattern = os.path.join(data_dir, 'wavs', '*.wav')
    # max_wav_length = find_max_wav_len(glob.glob(pattern)) + 50
    max_wav_length = 5150259
    print('Maximum wav length', max_wav_length)
    train_dir = os.path.join(data_dir, 'train', 'wavs')
    convert_wav(train_dir, max_wav_length)
    test_dir = os.path.join(data_dir, 'test', 'wavs')
    convert_wav(test_dir, max_wav_length)

