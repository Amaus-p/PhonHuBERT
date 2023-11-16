import os
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_checkpoint_dir(hparams):
  mode = 0o755
  # datetime object containing current date and time
  now = datetime.now()
  # checkpoint_dir = f'/home/yunkaiji/data2/note_transcription/dtmst/checkpoints_ap/{1}'.replace(' ', '_').replace(':', '-').replace('.', '-', 1)
  checkpoint_dir = hparams['checkpoint_dir'] + str(now).replace(' ', '_').replace(':', '-').replace('.', '-', 1)
  os.mkdir(checkpoint_dir, mode)
  return checkpoint_dir

def save_checkpoint(model, hparams, optimizer, epoch, train_loss, val_loss, checkpoint_name):  
    checkpoint = {  
        'model_state_dict': model.state_dict(),  
        'hparams': hparams,  
        'optimizer_state_dict': optimizer.state_dict(),  
        'epoch': epoch,  
        'trainign_loss': train_loss,
        'validation_loss': val_loss
    }  
    # checkpoint_path = os.path.join(hparams['checkpoint_dir'], checkpoint_name + '.pth')  
    checkpoint_path = os.path.join(checkpoint_name + '.pth')  
    torch.save(checkpoint, checkpoint_path)


def train_model_ctc(     
    hparams,
    device,
    train_loader, 
    val_loader,
    model,
    optimizer,
):
    counter = 0
    valid_loss_min = np.Inf
    ep_min = 0
    model.train()  
    checkpoint_dir = create_checkpoint_dir(hparams)
    print(checkpoint_dir)
    for ep in range(hparams['epochs']):
        hc = model.init_hidden(hparams['batch_size'])
        for inputs, onset_target, phon_target, ctc_target, _ in train_loader:
            counter += 1
            model.zero_grad()
            inputs, phon_target, onset_target, ctc_target = inputs.to(device), phon_target.to(device), onset_target.to(device), ctc_target.to(device)
            onset_out, phon_out, hc, losses = model.run_on_batch(inputs, hc, onset_target, phon_target, ctc_target, ep)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), hparams['clip'])
            optimizer.step() 
            if counter%hparams['print_every'] == 0:
                val_hc = model.init_hidden(hparams['batch_size'])
                val_losses_lst = []
                model.eval()
                for inputs, onset_target, phon_target, ctc_target, _ in val_loader:
                    inputs, phon_target, onset_target, ctc_target = inputs.to(device), phon_target.to(device), onset_target.to(device), ctc_target.to(device)
                    val_onset_out, val_phon_out, val_hc, val_losses = model.run_on_batch(inputs, val_hc, onset_target, phon_target, ctc_target, ep)
                    val_losses_lst.append(val_losses["total"].item())    
                model.train()
                print("Epoch: {}/{}...".format(ep+1, hparams['epochs']),
                    "Step: {}...".format(counter),
                    "Phon CE Loss: {:.6f}...".format(losses["phon_ce_loss"].item()),
                    "Onset CE Loss: {:.6f}...".format(losses["onset_ce_loss"].item()),
                    "CTC Loss: {:.6f}...".format(losses["ctc_loss"].item()),
                    "Loss: {:.6f}...".format(losses["total"].item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses_lst)),
                    "Current min val loss: {:.6f}".format(valid_loss_min),
                    "Epoch of min val loss: {}".format(ep_min),
                    f"| Ckp dir: {checkpoint_dir.split('/')[-1]}")
                if np.mean(val_losses_lst) < valid_loss_min:
                    save_checkpoint(model, hparams, optimizer, ep, losses, val_losses_lst, f'{checkpoint_dir}/val_total')
                    # torch.save(model.state_dict(), f'./model_save/state_dict_{base_name}.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, np.mean(val_losses_lst)))
                    valid_loss_min = np.mean(val_losses_lst)   
                    ep_min = ep+1
    return model  


def train_model(        
        hparams,
        device,
        train_loader, 
        val_loader,
        model,
        criterion,
        optimizer,
        base_name):
    counter = 0
    valid_loss_min = np.Inf
    ep_min = 0
    model.train()  
    if hparams["scheduler"]:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=hparams['scheduler_patience'], verbose=True)

    checkpoint_dir = create_checkpoint_dir(hparams)

    print(checkpoint_dir)
    for ep in range(hparams['epochs']):
        if hparams['hidden']:
            hc = model.init_hidden(hparams['batch_size'])
            for inputs, labels, _ in train_loader:
                if hparams['use_same_dim_pred_target']:
                    labels = labels.permute(0,2,1)
                counter += 1
                model.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                output, hc = model(inputs, hc)
                tansposed_output = torch.transpose(output[:,:,:60], 1, 2)

                new_loss = criterion(tansposed_output, labels)

                if hparams['use_cross_entropy']:
                    loss = new_loss
                else :
                    loss = torch.sum(new_loss)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), hparams['clip'])
                optimizer.step() 
                if counter%hparams['print_every'] == 0:
                    val_h = model.init_hidden(hparams['batch_size'])
                    val_losses = []
                    model.eval()
                    for inp, lab, _ in val_loader:
                        inp, lab = inp.to(device), lab.to(device)
                        out, val_h = model(inp, val_h)
                        tansposed_out = torch.transpose(out[:,:,:60], 1, 2)
                        if hparams['use_same_dim_pred_target']:
                            lab = lab.permute(0,2,1)
                        new_val_loss = criterion(tansposed_out, lab)
                        if hparams['use_cross_entropy']:
                            val_loss = new_val_loss
                        else :
                            val_loss = torch.sum(new_val_loss)
                        val_losses.append(val_loss.item())    
                    model.train()
                    print("Epoch: {}/{}...".format(ep+1, hparams['epochs']),
                        "Step: {}...".format(counter),
                        "Loss: {:.6f}...".format(loss.item()),
                        "Val Loss: {:.6f}".format(np.mean(val_losses)),
                        "Current min val loss: {:.6f}".format(valid_loss_min),
                        "Epoch of min val loss: {}".format(ep_min),
                        f"| Ckp dir: {checkpoint_dir.split('/')[-1]}")
                    if hparams["scheduler"]:
                        scheduler.step(np.mean(val_losses))
                    if np.mean(val_losses) < valid_loss_min:
                        save_checkpoint(model, hparams, optimizer, ep, loss, val_losses, f'{checkpoint_dir}/{base_name}')
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                        valid_loss_min = np.mean(val_losses)   
                        ep_min = ep+1
        else:
            for inputs, labels, _ in train_loader:
                counter += 1
                model.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                tansposed_output = torch.transpose(output, 1, 2)

                new_loss = criterion(tansposed_output, labels)
                if pd.isna(new_loss):
                    print('Loss is diverging, breaking the training')
                    return model
                else:
                    loss = new_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), hparams['clip'])
                optimizer.step() 

                if counter%hparams['print_every'] == 0:
                    val_losses = []
                    model.eval()
                    for inp, lab, _ in val_loader:
                        inp, lab = inp.to(device), lab.to(device)
                        out = model(inp)
                        tansp_out = torch.transpose(out, 1, 2)
                        val_loss = criterion(tansp_out, lab)
                        val_losses.append(val_loss.item())
                    model.train()
                    print("Epoch: {}/{}...".format(i+1, hparams['epochs']),
                        "Step: {}...".format(counter),
                        "Loss: {:.6f}...".format(loss.item()),
                        "Val Loss: {:.6f}".format(np.mean(val_losses)),
                        f"| Ckp dir: {checkpoint_dir.split('/')[-1]}")
                    if np.mean(val_losses) < valid_loss_min:
                        save_checkpoint(model, hparams, optimizer, ep, loss, val_losses, f'{checkpoint_dir}/{base_name}')
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                        valid_loss_min = np.mean(val_losses) 
    return model  