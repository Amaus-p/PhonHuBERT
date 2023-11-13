#libraries
import torch
import torch.nn as nn 

import pandas as pd
import numpy as np
import random as rd

from sklearn.feature_extraction.text import CountVectorizer

#locql imports
from local_attention import LocalAttention
import torch.nn.functional as F

#https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/


class LSTM(nn.Module):
    def __init__(self, hparams, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.hidden_dim = hparams['input_dim']// hparams['n_layers'] if hparams['new_new_model'] else hparams['hidden_dim']
        # self.hidden_dim = hparams['hidden_dim']
        self.n_layers = hparams['n_layers']
        self.batch_size = hparams['batch_size']
        self.seq_len = hparams['seq_length']
        self.bidirectional = hparams['bidirectional']
        self.use_ctc_onset = hparams['use_ctc_onset']
        self.device = device
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=hparams['batch_first'], bidirectional=self.bidirectional)
        self.fully_connected = nn.Linear(self.hidden_dim * (1+self.bidirectional), self.output_dim)
        self.dropout = nn.Dropout(hparams['drop_prob'])
    
    def forward(self, x, hc):
        hc = tuple([e.data for e in hc])
        lstm_out, hc = self.lstm(x, hc)
        out = self.fully_connected(lstm_out)
        out = self.dropout(out)
        return out, hc
    
    def init_hidden(self, batch_size):
        return (
            torch.zeros((1+self.bidirectional)*self.n_layers, batch_size, self.hidden_dim, device=self.device), 
            torch.zeros((1+self.bidirectional)*self.n_layers, batch_size, self.hidden_dim, device=self.device)
            )

class LSTMCTCOnset(nn.Module):
    def __init__(self, hparams, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.hidden_dim = hparams['input_dim']// hparams['n_layers'] if hparams['new_new_model'] else hparams['hidden_dim']
        self.n_layers = hparams['n_layers']
        self.batch_size = hparams['batch_size']
        self.seq_len = hparams['seq_length']
        self.bidirectional = hparams['bidirectional']
        self.device = device
        # self.lstm1 = nn.LSTM(input_dim, output_dim, n_layers)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=hparams['batch_first'], bidirectional=self.bidirectional)
        # self.lstm2 = nn.LSTM(hidden_dim, output_dim, n_layers)
        
        self.fc1 = nn.Linear(self.hidden_dim * (1+self.bidirectional), self.hidden_dim //2)
        self.fc2 = nn.Linear(self.hidden_dim //2, self.hidden_dim //8)
        self.fc3 = nn.Linear(self.hidden_dim //8, 3)
        self.dropout = nn.Dropout(hparams['drop_prob'])
    
    def forward(self, x, hc):
        hc = tuple([e.data for e in hc])
        out, hc = self.lstm(x, hc)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out, hc
    
    def init_hidden(self, batch_size):
        return (
            torch.zeros((1+self.bidirectional)*self.n_layers, batch_size, self.hidden_dim, device=self.device), 
            torch.zeros((1+self.bidirectional)*self.n_layers, batch_size, self.hidden_dim, device=self.device)
            )
    
class CNNONset(nn.Module):
    def __init__(self, hparams, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.t_conv1 = nn.ConvTranspose2d(
            in_channels=hparams['input_dim'],
            out_channels=hparams['input_dim']//2,
            kernel_size=9,
        )
        self.t_conv2 = nn.ConvTranspose2d(
            in_channels=hparams['input_dim']//2,
            out_channels=hparams['input_dim']//4,
            kernel_size=9,
        )
        self.t_conv3 = nn.ConvTranspose2d(
            in_channels=hparams['input_dim']//4,
            out_channels=hparams['input_dim']//8,
            kernel_size=9,
        )
        self.t_conv4 = nn.ConvTranspose2d(
            in_channels=hparams['input_dim']//8,
            out_channels=hparams['input_dim']//16,
            kernel_size=9,
        )
        self.fc1 = nn.Linear(hparams['input_dim']//16, 3)
        self.fc2 = nn.Linear(hparams['input_dim']*33, )
        self.dropout = nn.Dropout(hparams['drop_prob'])
    
    def forward(self, x):
        x = x.unsqueeze_(1)
        x = x.permute(0, 3, 2, 1)
        print(x.shape)
        x = torch.relu(self.t_conv1(x))
        print(x.shape)

        x = torch.relu(self.t_conv2(x))
        print(x.shape)
        x = torch.relu(self.t_conv3(x))
        print(x.shape)
        x = torch.relu(self.t_conv4(x))
        x = torch.flatten(x, 2, 3)

        print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        return x

class LSTMCTCPhon(nn.Module):
    def __init__(self, hparams, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.hidden_dim = hparams['input_dim']// hparams['n_layers'] if hparams['new_new_model'] else hparams['hidden_dim']
        self.n_layers = hparams['n_layers']
        self.batch_size = hparams['batch_size']
        self.seq_len = hparams['seq_length']
        self.bidirectional = hparams['bidirectional']
        self.device = device
        # self.lstm1 = nn.LSTM(input_dim, output_dim, n_layers)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=hparams['batch_first'], bidirectional=self.bidirectional)
        # self.lstm2 = nn.LSTM(hidden_dim, output_dim, n_layers)
        
        self.fc1 = nn.Linear(self.hidden_dim * (1+self.bidirectional), self.hidden_dim //4)
        self.fc2 = nn.Linear(self.hidden_dim //4, self.output_dim)
        self.dropout = nn.Dropout(hparams['drop_prob'])
    
    def forward(self, x, hc):
        hc = tuple([e.data for e in hc])
        out, hc = self.lstm(x, hc)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        return out, hc
    
    def init_hidden(self, batch_size):
        return (
            torch.zeros((1+self.bidirectional)*self.n_layers, batch_size, self.hidden_dim, device=self.device), 
            torch.zeros((1+self.bidirectional)*self.n_layers, batch_size, self.hidden_dim, device=self.device)
            )
    
class CTC_CE_Loss(nn.Module):
    def __init__(self, hparams, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hparams=hparams
        self.device = device
        if self.hparams['model_name'] == 'CNN':
            self.onset_model = CNNONset(hparams, device)
        else:
            self.onset_model = LSTMCTCOnset(hparams, device)
        self.phon_model = LSTMCTCPhon(hparams, device)

    def forward(self, x, onset_hc, phon_hc):
        if self.hparams['model_name'] == 'CNN':
            onset_out = self.onset_model(x)
        else:
            onset_out, onset_hc = self.onset_model(x, onset_hc)
        phon_out, phon_hc = self.phon_model(x, phon_hc)
        return onset_out, phon_out, onset_hc, phon_hc
    
    def init_hidden(self, batch_size):
        phon_hc = self.phon_model.init_hidden(batch_size)
        if not self.hparams['model_name'] == 'CNN':
            onset_hc = self.onset_model.init_hidden(batch_size)
            return [onset_hc, phon_hc]
        return [phon_hc]
    
    def infer(self, x, HC):
        if self.hparams['model_name'] == 'CNN':
            phon_hc = HC[0]
        else:
            [onset_hc, phon_hc] = HC
        onset_out, phon_out, onset_hc, phon_hc = self.forward(x, onset_hc, phon_hc)
        return torch.sigmoid(onset_out), torch.softmax(phon_out, dim=2), onset_hc, phon_hc
    
    def run_on_batch(self, x, HC, onset_target, phon_target, ctc_target, ep):
        if self.hparams['model_name'] == 'CNN':
            phon_hc = HC[0]
            onset_out, phon_out, phon_hc = self.forward(x, None, phon_hc)
        else:
            [onset_hc, phon_hc] = HC         
            onset_out, phon_out, onset_hc, phon_hc = self.forward(x, onset_hc, phon_hc)
       
        onset_ce_loss = F.binary_cross_entropy(torch.sigmoid(onset_out[:,:,1]), onset_target.float())
        phon_ce_loss = F.cross_entropy(phon_out.permute(0,2,1), phon_target)
        ctc_loss = torch.tensor(100)

        if ep >= self.hparams['ep_start_ctc']:

            expand_out_phon = torch.ones(1, phon_out.shape[1], 1).to(self.device)
            expand_first_channel_pitchs = expand_out_phon.repeat(1, 1, 2).to(self.device)*(0.5)
            phon_out = torch.cat((expand_first_channel_pitchs, phon_out), dim=2)

            last_channel_onsets = onset_out[:, :, -1:]  #we take the last channel of the onset data
            expand_outputs_onsets = last_channel_onsets.repeat(1, 1, 59).to(self.device) # the last channel is the probability of having a pitch it will be multiplied to the probabilities potential pitches
            onset_out = torch.cat((onset_out, expand_outputs_onsets), dim=2)

            output_ctc = F.log_softmax(phon_out * onset_out, dim=2)

            ctc_loss = F.ctc_loss(output_ctc.permute(1,0,2), ctc_target, torch.tensor([output_ctc.shape[1]]*output_ctc.shape[0]).to(self.device), torch.tensor([ctc_target.shape[1]]*ctc_target.shape[0]).to(self.device))
        total_loss = 1/2*onset_ce_loss + 1/2*phon_ce_loss + ctc_loss
        losses = {"total": total_loss, "onset_ce_loss":onset_ce_loss, "phon_ce_loss":phon_ce_loss, "ctc_loss":ctc_loss}

        return onset_out, phon_out, [onset_hc, phon_hc] if not self.hparams['model_name'] == 'CNN' else [phon_hc], losses
