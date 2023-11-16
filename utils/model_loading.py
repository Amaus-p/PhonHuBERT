import torch
import torch.nn as nn
import torchvision
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
    elif model_name=='double_LSTM':
        model = md.DOUBLE_LSTM(hparams, device).to(device)
    else:
        raise Valueprint('MODEL NOT FOUND')
    if hparams['use_cross_entropy']:
        criterion = nn.CrossEntropyLoss()
    elif hparams['use_focal_loss']:
        criterion = torchvision.ops.sigmoid_focal_loss
    elif hparams['use_mse_loss']:
        criterion = nn.MSELoss()
    print("CRITERION", criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    return model, criterion, optimizer
