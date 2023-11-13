import torch
import torch.nn as nn
from . import models as md

def load_model(
        device,
        hparams,
        ):
    model_name = hparams['model_name']
    print('model_name', model_name)
    if model_name=='LSTM':
        if hparams['use_ctc_onset']:
            model = md.CTC_CE_Loss(hparams, device).to(device)
        else:
            model = md.LSTM(hparams, device).to(device)
    else:
        raise Valueprint('MODEL NOT FOUND')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    return model, criterion, optimizer
