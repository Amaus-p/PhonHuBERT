#libraries
from torch.utils.data import DataLoader
import numpy as np
import random as rd

from . import create_label_tools as clt

class MusicDataLoader():
    def __init__(self, 
        hparams,
        is_test=False,
    ) -> None:
        """
        Input:
            dataset_name:
                name of the dataset, for the moment only :
                    - opencpop
                    - m4singer
                    are managed
            data_root:
                directory where all the audio wav data and textGrid data are located
            segmented:
                True: the wavs data will segmented of a lenth of env. 5 to 10 sec. We would need to use some batch for the learning
                False: the wav data are not segmented in small parts. No batch for the learning
            padding :
                True : I will padd all the segments so that they are of the same length
                False : I won't padd the segments
            batch_size :
                Size of the batches we will use
        Output:
            None
        """
        self.supported_datasets = [
            'opencpop',
        ]
        if hparams['dataset_name'] not in self.supported_datasets:
            raise ValueError('This dataset is not supported')
        
        rd.seed(hparams['seed'])
        self.dataset_name = hparams['dataset_name']
        self.data_root = hparams['data_root_dir']
        self.segmented = hparams['segmented']
        self.padding = hparams['padding']
        self.batch_size = hparams['batch_size']
        self.hparams = hparams
        self.is_test = is_test
        self.train_test_folder = '/train/' if not is_test else '/test/'
        self.use_ctc_onset = hparams['use_ctc_onset']

        if hparams['dataset_name']=='opencpop':
            self.frames_per_sec = 50
            self.silence_phon = 'SP'
            if self.segmented:
                self.tiers_to_get=['音素', '句子']
                self.wav_folder=self.train_test_folder + '/segments/wavs/'
            else:
                self.wav_folder= self.train_test_folder + '/wavs/'
                self.tiers_to_get=['音素']
            self.textGrid_folder= self.train_test_folder + "/textgrids"

    def convert_wav_to_hidden(self):        
        clt.convert_wav_to_hidden(
            wav_folder=self.data_root + self.wav_folder,
            padding=self.padding
        )

    def main_create_phon_set(self):
        """
        Output:
            train_loader:
                Dataloader of batch_size = self.batch_size with train data
            val_loader:
                Dataloader of batch_size = self.batch_size with val data
            test_loader: 
                Dataloader of batch_size = self.batch_size with test data
            phon_list:
                List of phonemes  
            embedded_phonemes:
                Dictionnary of key = phoneme and value = embedding
        Return train, val and test loaders
        """
        translated_labeled_data, phon_list, embedded_phonemes = clt.create_labels_for_hidden(
            hparams=self.hparams,
            textgrid_folder= self.data_root + self.textGrid_folder,
            tiers_to_get=self.tiers_to_get,
            segmented=self.segmented,
            silence_phon=self.silence_phon,
            frames_per_sec=self.frames_per_sec,
            dataset_name=self.dataset_name
        )
        print('End creation of labels ')
        hidden_space_data = clt.load_hidden_space_data(
            wav_folder=self.data_root + self.wav_folder,
            dataset_name=self.dataset_name,
            hparams=self.hparams,
        ) 
        print('End get hidden_space_data')
        labeled_data = clt.add_label_to_data(
            translated_labeled_data, 
            hidden_space_data, 
            phon_list,
            silence_phon=self.silence_phon,
            silence_emb = embedded_phonemes[self.silence_phon],
            is_test = self.is_test,
            ctc_model = self.use_ctc_onset,
            )
        print('End labeling data')
        if not self.is_test:
            val, train = clt.split_data(
                labeled_data,
                val_split_rate=0.1,
                )
            
            print("Validation data length: ", len(val))
            print("Training data length: ", len(train))

            train_loader = DataLoader(train, batch_size=self.batch_size)
            val_loader = DataLoader(val, batch_size=self.batch_size)
            length_sampled_vectors = np.max(np.array([len(e[1]) for e in train]))
            return train_loader, val_loader, phon_list, embedded_phonemes, length_sampled_vectors
        test_loader = DataLoader(labeled_data, batch_size=1)

        return test_loader, phon_list, embedded_phonemes
