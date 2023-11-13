import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocessing.hubertinfer import HubertEncoder
from utils.infer_utils import get_end_file
import librosa

data_dir = "../data/opencpop"

def find_max_wav_len(wav_paths):
    max_length = 0
    step = 0
    max_step = len(wav_paths)
    for wav_path in wav_paths:
        print(f'\r step/number of step : {step}/{max_step}', end='')
        wav, sr = librosa.load(wav_path, sr=None)
        wav16 = librosa.resample(wav, sr, 16000) if sr != 16000 else wav
        length = len(wav16)
        if length>max_length:
            max_length=length
        step+=1
    return max_length

def convert_wav(path=data_dir + "/segments/wavs", m4singer=False):
    """
    Taken from the file pre-hubert.py
    I don't use the function itself to be able to manipulate it

    I MODIFY EVERY NPY FILE SO THAT they are all of the same length by adding silences at the end of the wavs files
    """
    print(path)
    hubert_model = HubertEncoder(hubert_mode='soft_hubert')
    wav_paths = get_end_file(path, ".wav")
    max_len = find_max_wav_len(wav_paths) + 100
    print(max_len)
    with tqdm(total=len(wav_paths)) as p_bar:
        p_bar.set_description('Processing HuBERT hidden space vectors')
        for wav_path in wav_paths:
            npy_path = Path(wav_path).with_suffix(".npy")
            if not os.path.exists(npy_path):
                np.save(str(npy_path), hubert_model.encode(wav_path, max_len))
            p_bar.update(1)
