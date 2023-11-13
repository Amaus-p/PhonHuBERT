import numpy as np
import random as rd
import os
from pathlib import Path
from praatio import textgrid
from tqdm import tqdm
import librosa
import torch
import fairseq

import sys
import glob

sys.path.insert(0, os.path.abspath('../'))
#local imports
from preprocessing.hubertinfer import HubertEncoder
from utils.infer_utils import get_end_file

       
def find_max_wav_len(wav_paths: list[str]):
    """
    Input
        wav_paths: 
            list of path names of the wav file we want to study

    output
        max_length:
            the length of the longest wav file
    """
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
    return(max_length)

def convert_wav_to_hidden(wav_folder:str, padding=True, isHubert=True):        
    """
    Taken from the file pre-hubert.py

    Input: 
        wav_folder:
            the path to the wav folder
        padding:
            True: we add some padding to unify all the length
            False: we don't padd

    Ouput:
        No output but create .npy files for every .wav file in the wav folder
    """
    # hubert_mode可选——"soft_hubert"、"cn_hubert"
    hubert_model = HubertEncoder(hubert_mode='soft_hubert') 
    wav_paths = get_end_file(wav_folder, "wav")
    max_len = find_max_wav_len(wav_paths) if padding else -1
    # max_len = 1119839
    print('\n', max_len)
    with tqdm(total=len(wav_paths)) as p_bar:
        p_bar.set_description('Processing')
        for wav_path in wav_paths:
            npy_path = Path(wav_path).with_suffix(".npy")
            if not os.path.exists(npy_path):
                np.save(str(npy_path), hubert_model.encode(wav_path, max_len))
            p_bar.update(1)

def remove_files(path, extension):
    paths_to_remove = get_end_file(path, extension)
    with tqdm(total=len(paths_to_remove)) as p_bar:
        p_bar.set_description('Processing')
        for file in paths_to_remove:
            # print(file)
            os.remove(file)
            p_bar.update(1)
    return

def get_name(path: str, dataset_name: str, to_replace = '.npy'):
    """
    Get the name of the audio we are looking at

    Input:
        path:
            Path of the concerned file
        dataset_name:
            Name of the dataset because we have different management of the name depending of the name of the dataset
        to_replace:
            Extension of the file we are looking at. It's the part of the name that we will remove

    Output:
        name of the audio we are looking at
    """
    if dataset_name=='opencpop':
        return path.split('/')[-1].replace(to_replace,'')
    elif 'm4singer' in dataset_name:
        split_path = path.split('/')
        return split_path[-2] + split_path[-1].replace(to_replace,'')
    else:
        raise ValueError('This dataset is not managed')

def create_name_segment(segment_number: int, audio_name: str):
    """
    Concatenate the segment number and the audio name to have the same format than the ones in the segments folder
    """
    seg = str(segment_number)
    for _ in range(6 - len(seg)):
        seg = '0' + seg
    return audio_name+seg

def create_phone_segments(phonemeTier, segmentsTier, segment_number: int, phon_set: set):
    """
    This function has the purpose of creating the segments of phonemes and duration of phoneme for every segments
    We used the TextGrid files to do that.

    Indeed the segmentsTier is the tier that contains the segmentation of the lyrics like we have in the folder segments of the openCPOP dataset.
    The phonemeTier is the tier that contains all the phonemes of the lyrics with starting and ending time but there is no sub-segmentation in the segments defined by segmentsTier.
    That's what we want to do here: take these phonemes and but them into the segmentation of the segmentsTier.

    Input: 
        phonemeTier
        segmentsTier
        segment_number=0 # every audio increase the segment_number by 1 comparing to the segment number of the last segment of the previous audio.
    Output: 
        segments :  
            a dictionnary of key 'segment_number' 
            and of value a list of tupples of 3 elements 
                (starting_phoneme_time, ending_phoneme_time, phoneme)
    """
    phoneme_number = 0
    segments = {}
    start_phon, end_phon, phon = phonemeTier.entries[phoneme_number]
    for start_sec, end_sec, section in segmentsTier.entries:
        if section!='silence':
            segment_number+=1
            segments[segment_number]=[]
            while end_phon <= start_sec:
                phoneme_number+=1
                start_phon, end_phon, phon = phonemeTier.entries[phoneme_number]
            while start_phon < end_sec:
                start = max(start_phon, start_sec)
                end = min(end_phon, end_sec)            
                segments[segment_number].append((start-start_sec,end-start_sec,phon))
                phon_set.add(phon)
                if end_phon > end_sec:
                    start_phon=end_sec
                    break
                else: 
                    phoneme_number+=1
                    start_phon, end_phon, phon = phonemeTier.entries[phoneme_number]
    return segments, segment_number

def create_phone_segments_for_every_audio(tier_list: list, segmented: bool, dataset_name: str):
    """
    Input:
        tier_list:
            List of tuple (name_audio, tier1, tier2...)
        segmented:
            if we segment the big audios into small parts or not
        dataset_name:
            name of the dataset we are using. The management of the data is different from one dataset to another
    
    Output:

    """
    segment_number = 0
    start_stop_phon = {}
    phon_set = set()
    max_duration = 0
    if segmented and dataset_name=='opencpop':
        for audio_name, phonemeTier, segmentsTier in tier_list:
            segments, segment_number = create_phone_segments(phonemeTier, segmentsTier, segment_number, phon_set)
            for seg in segments:
                start_stop_phon[create_name_segment(seg, audio_name)] = segments[seg]
                if segments[seg][-1][1] > max_duration:
                    max_duration=segments[seg][-1][1]
    else:
        for audio_name, tier_phonemes in tier_list:
            start_stop_phon[audio_name]=[]
            for start, stop, label in tier_phonemes.entries:
                start_stop_phon[audio_name].append((start, stop, label))
                if stop > max_duration:
                    max_duration=stop
                phon_set.add(label)
    return start_stop_phon, list(phon_set), max_duration

def extract_text_grid_data(textgrid_folder: str, tiers_to_get: list[str], segmented: bool, dataset_name: str):
    """
    https://nbviewer.org/github/timmahrt/praatIO/blob/main/tutorials/tutorial1_intro_to_praatio.ipynb
    Function to access textgrid data.

    Input: 
        textgrid_folder:
            path to the textgrid directory
        tiers_to_get:
            names of the tier we want to get
        segmented:
            if we segment the big audios into small parts or not
        dataset_name:
            name of the dataset we are using. The management of the data is different from one dataset to another

    Output:
        start_stop_phon:
            dictionary of
                key: name of the audio/segment
                value: list of tuples (start, end, phon) with:
                    start: the starting time of the phoneme
                    end: the ending time of the phoneme
                    phon: the phoneme
        a list of all the phonemes in the dataset. It will be used for embedding.
        max_duration:
            the duration of the longuest segment
    """
    txt_grid_paths = get_end_file(textgrid_folder, "TextGrid")
    tier_list = []
    for path in txt_grid_paths:
        name = get_name(path, dataset_name, ".TextGrid")
        tg = textgrid.openTextgrid(path, includeEmptyIntervals=False ,duplicateNamesMode='rename')
        if segmented and dataset_name=='opencpop':
            phonemeTier = tg.getTier(tiers_to_get[0])
            segmentsTier = tg.getTier(tiers_to_get[1])
            tier_list.append((name, phonemeTier, segmentsTier))
        else:
            tier_phonemes = tg.getTier(tiers_to_get[0])
            tier_list.append((name, tier_phonemes))
    return create_phone_segments_for_every_audio(sorted(tier_list), segmented, dataset_name)

def create_set_phonemes_and_embedding(phoneme_list: list[str]):
    """
    https://towardsdatascience.com/simple-word-embedding-for-natural-language-processing-5484eeb05c06
    Input:
        phoneme_list:
            a list of all the phonemes of the dataset
    Ouput: 
        a dictionary to reverse the key-value pairs of the list for phoneme embedding.
        Example is phoneme_list[10]=='v' therfore dic['v']==10
    """
    dic = {}
    i = 0
    for phon in phoneme_list:
        dic[phon] = i
        i+=1
    return dic

def unify_length(segments: dict[str, list[tuple[float, float, str]]], max_duration: float, silence_phoneme: str):
    """
    Function to padd all the labelled data.

    Input:
        segments:
            All the segments of
                key: name of the segment
                value: list of tuples of (start, end, phon)
        max_duration:
            Duration of the longuest segment
        silence_phoneme:
            Value used for padding/

    Output:
        segments:
            Same type of the input but where we added the padded values
    """
    for _, seg in segments.items():
        seg.append((seg[-1][1], max_duration, silence_phoneme))
    return segments

def sampling(segments: dict[str, list[tuple[float, float, str]]], frames_per_sec = 106):
    """
    This function has the purpose of splitting the time for every sequence into frames 

    Input:
        segments:
            All the segments of
                key: name of the segment
                value: list of tuples of (start, end, phon)
        frames_per_sec:
            Number of frame in one second

    Output: 
        framed_segments:
            Segments of 
                key: name of the segment
                value: list of tuples of (start, end, phon) where a lot of elements are duplicated
                    according to the frequency rate
    
            In framed_data I put for every frame, the phoneme that corresponds to the time of this frame. 
            If in the frame, there are 2 phonemes, I that the one that has the longest duration in the frame

    segments = {
        1: [(start, end, phon),(start, end, phon),(start, end, phon),...],
        2: [(start, end, phon),(start, end, phon),(start, end, phon),...],
        ...
    }
    """
    framed_segments = {}
    frame_time = 1/frames_per_sec
    for seg in segments:
        T_max = segments[seg][-1][1]
        # In framed_data I put for every frame, the phoneme that corresponds to the time of this frame. 
        # If in the frame, there are 2 phonemes, I that the one that has the longest duration in the frame
        framed_data = []
        start_frame = 0
        end_frame = start_frame+frame_time
        phon_number = 0
        total_phon_number = len(segments[seg])
        while end_frame<T_max and phon_number < total_phon_number:
            _, end_phon, phon = segments[seg][phon_number]
            if end_frame <= end_phon :
                framed_data.append(phon)
                start_frame = end_frame
                end_frame+=frame_time
            elif start_frame < end_phon:
                if end_phon-start_frame > frame_time/2:
                    framed_data.append(phon)
                    phon_number+=1
                    start_frame = end_frame
                    end_frame+=frame_time
                else:
                    phon_number+=1
                    framed_data.append(phon)
                    start_frame = end_frame
                    end_frame+=frame_time
            elif start_frame >= end_phon:
                phon_number+=1
        framed_segments[seg] = framed_data
    return framed_segments

def translation_to_embedding_space(framed_segments: dict[str, list[tuple[float, float, str]]], embedded_phonemes: dict[str, int]):
    """
    Input: 
        framed_segments:
            Segments of 
                key: name of the segment
                value: list of tuples of (start, end, phon) where a lot of elements are duplicated
                    according to the frequency rate
        embedded_phonemes: 
            the translation of each phonemes in the embedded space: dic of key 'phon' and value the embedded phoneme
    
    Output: 
        the tranlation of each phoneme in every frame of the framed_semgents data into the embedded space
    """
    translated_segmentation = {}
    for seg_name, data in framed_segments.items():
        translated_segmentation[seg_name]=[]
        for phon in data:
            translated_segmentation[seg_name].append(embedded_phonemes[phon])
    return translated_segmentation

def create_labels_for_hidden(hparams, textgrid_folder: str, tiers_to_get: str, segmented: bool, silence_phon: str, frames_per_sec: int, dataset_name: str):
    """
    Main function for the creation of labels

    Input:
        textgrid_folder: 
            name of the folder to find textGrid data
        tiers_to_get:
            names of the tier we want to get
        segmented:
            if we segment the big audios into small parts or not
        silence_phon:
            element used for padding
        frames_per_sec:
            number of frames per second
        dataset_name:
            name of the dataset we are using. The management of the data is different from one dataset to another

    Output:
        translated_labeled_data:
            Dictionnay of
                key: name of the segment
                value: list of tuples (start, end, translated_to_embedding_space_phoneme) for every frame of the padded, framed segment
        phon_list:
            list of phonemes 
        embedded_phonemes:
            dictionnary of reverse embedded phonemes
    """
    data_root = hparams['data_root_dir']
    phon_list_path = data_root + 'phon_list.npy'
    embedded_phonemes_path = data_root + 'embedded_phonemes.npy'
    if os.path.exists(phon_list_path):
        print("loading the phoneme list: ", phon_list_path)
        phon_list = np.load(phon_list_path, allow_pickle=True)
        start_stop_phon, _, _ = extract_text_grid_data(textgrid_folder, tiers_to_get, segmented, dataset_name)
    else:     
        print("The phoneme list doesn't exist, creating it")
        start_stop_phon, phon_list, _ = extract_text_grid_data(textgrid_folder, tiers_to_get, segmented, dataset_name)
        np.save(phon_list_path, phon_list)
    if os.path.exists(embedded_phonemes_path):
        print("loading the embedded phonemes list: ", embedded_phonemes_path)
        embedded_phonemes = np.load(embedded_phonemes_path, allow_pickle=True).item()
    else:
        print("The embedded phonemes list doesn't exist, creating it")
        embedded_phonemes = create_set_phonemes_and_embedding(phon_list)
        np.save(embedded_phonemes_path, embedded_phonemes)
    unify_length(start_stop_phon, hparams["max_length"], silence_phon)
    framed_segments = sampling(start_stop_phon, frames_per_sec)
    translated_labeled_data = translation_to_embedding_space(framed_segments, embedded_phonemes)
    return translated_labeled_data, phon_list, embedded_phonemes

def load_hidden_space_data(wav_folder:str, dataset_name:str, hparams: dict):
    """
    Input: 
        wav_folder:
            path of the directory containing the vectors of the hidden space created by huBERT
        dataset_name:
            name of the datset
    Output: 
        dictionnary of 
            key: segment name
            value: vectors contained in the npy file generated by huBERT
    """
    print('WAV folder, loading the hidden space data', wav_folder)
    npy_paths = glob.glob(wav_folder + '*npy')
    # wav_paths = get_end_file(wav_folder, "npy")
    print('Number of files', len(npy_paths))
    data = {}
    with tqdm(total=len(npy_paths)) as p_bar:
        p_bar.set_description('Processing')
        for path in npy_paths:
            if hparams['load_audio_plus_hubert']:
                x, sr = librosa.load(path.replace('.npy', '.wav'), sr=None)
                data[get_name(path, dataset_name, '.npy')] = [np.load(path, ), x]
            else:
                data[get_name(path, dataset_name, '.npy')] = np.load(path, )
            p_bar.update(1)
    return data

def create_onset_ctc_list(phon_seq:list, max_ctc_length:int):
    ctc_target = [phon_seq[0]+2]
    onsets_target = np.zeros(len(phon_seq))
    for i in range(1,len(phon_seq)):
        phon_i = phon_seq[i]+2
        if phon_i!=ctc_target[-1]:
            ctc_target.append(1)
            ctc_target.append(phon_i)
            onsets_target[i] = 1
        if len(ctc_target)>max_ctc_length:
            max_ctc_length = len(ctc_target)
    return ctc_target, onsets_target, max_ctc_length

def add_label_to_data(translated_labeled_data: dict, hidden_space_data: dict, phon_list: list, silence_phon:str, silence_emb:int, is_test: bool = False, ctc_model = False):
    """
    Input: 
        take the segmeted labeled data audios 
        and the data from the hidden space 
    Add labels to the hidden space vectors from huBERT in order to use them in the RNN / CNN ...

    Output: 
        labeled data of the hidden space : 
        train_data : list of [data, label]
    I don't keep the name of the segment in memory
    """
    train_data = []
    max_ctc_length = 0
    print(silence_phon, silence_emb)
    assert silence_emb == 55, f'The silence embedding is not the one expected. It is {silence_emb} instead of 55'
    for segment_name, matrix in hidden_space_data.items():
        assert len(matrix) == len(translated_labeled_data[segment_name]), f"The length of the hidden space is of {len(matrix)}, while the length of the translated_labeled_data is of {len(translated_labeled_data[segment_name])}. {segment_name}"
        if is_test:
            gt = translated_labeled_data[segment_name]
            start = 0
            end = len(gt)
            for j, phon_i in enumerate(gt):
                if phon_list[phon_i]!=silence_phon and not start:
                    start = max(j-1, 0)
                if phon_list[phon_i]!=silence_phon:
                    end = j+1
            translated_labeled_data[segment_name] = gt[start:end]
            matrix = matrix[start:end]
        if ctc_model:
            ctc_target, onsets_lst, max_ctc_length = create_onset_ctc_list(translated_labeled_data[segment_name], max_ctc_length)
            train_data.append([np.array(matrix), onsets_lst, np.array(translated_labeled_data[segment_name]), ctc_target, segment_name])
        else:
            train_data.append([np.array(matrix), np.array(translated_labeled_data[segment_name]), segment_name])
    if ctc_model:
        for i in range(len(train_data)):
            train_data[i][3] = np.pad(train_data[i][3], (0, max_ctc_length-len(train_data[i][3])), 'constant', constant_values=(0, silence_emb+2))
    # print(np.array(train_data).shape)
    # print(train_data[0][2].shape)
    # with open('ctc.txt', 'w') as f:
    #     for i in range(len(train_data[0][2])):
    #         f.write(f'{train_data[0][2][i]}\n')
    # f.close()
    # exit()
    return train_data

def split_data(data: dict, val_split_rate: int):
    """
    Function to split the data into train, test and validation
    Returns test, val and train datasets
    """
    val_size = int(len(data)*val_split_rate)
    rd.shuffle(data)
    train_data = data[val_size:]
    val_data = data[:val_size]
    print(len(val_data))
    print(len(train_data))

    return val_data, train_data

# convert_wav_to_hidden_content_vec('/home/yunkaiji/data3/opencpop_contentVec/wavs', small = True)

# data = load_hidden_space_data('/home/yunkaiji/data3/m4singer_save', 'm4singer')
# print(data)
# for name, dat in data.items():
#     print(name, len(dat))


# remove_files('/home/yunkaiji/data3/opencpop_contentVec5', 'npy')
