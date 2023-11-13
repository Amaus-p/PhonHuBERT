import os.path
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from fairseq import fairseq

from modules.hubert.cn_hubert import load_cn_model, get_cn_hubert_units
from modules.hubert.hubert_model import hubert_soft, get_units
from modules.hubert.hubert_onnx import get_onnx_units

from utils_hubert.hparams import hparams

class HubertEncoder:
    def __init__(self, pt_path='checkpoints/hubert/hubert_soft.pt', hubert_mode='', onnx=False, contentVec = False):
        self.hubert_mode = hubert_mode
        self.onnx = onnx
        if 'use_cn_hubert' not in hparams.keys():
            hparams['use_cn_hubert'] = False
        if not contentVec:
            if hparams['use_cn_hubert'] or self.hubert_mode == 'cn_hubert':
                pt_path = "checkpoints/cn_hubert/chinese-hubert-base-fairseq-ckpt.pt"
                self.dev = torch.device("cuda:6")
                self.hbt_model = load_cn_model(pt_path)
            else:
                if onnx:
                    self.hbt_model = ort.InferenceSession("onnx/hubert_soft.onnx",
                                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider', ])
                else:
                    pt_path = list(Path(pt_path).parent.rglob('*.pt'))[0]
                    if 'hubert_gpu' in hparams.keys():
                        self.use_gpu = hparams['hubert_gpu']
                    else:
                        self.use_gpu = True
                    self.dev = torch.device("cuda:6" if self.use_gpu and torch.cuda.is_available() else "cpu")
                    self.hbt_model = hubert_soft(str(pt_path)).to(self.dev)
        else:
            if not (hparams['use_cn_hubert'] or self.hubert_mode == 'cn_hubert'):
                if not onnx:
                    if 'hubert_gpu' in hparams.keys():
                        self.use_gpu = hparams['hubert_gpu']
                    else:
                        self.use_gpu = True
            print("we use content vec")
            self.dev = torch.device("cuda:6" if self.use_gpu and torch.cuda.is_available() else "cpu")
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pt_path])
            self.hbt_model = models[0].to(self.dev)
        print(f"| load 'model' from '{pt_path}'")

    def encode(self, wav_path, max_wav_length=-1):
        if isinstance(wav_path, BytesIO):
            npy_path = ""
            wav_path.seek(0)
        else:
            npy_path = Path(wav_path).with_suffix('.npy')
        if os.path.exists(npy_path):
            print('not get-units-1')

            units = np.load(str(npy_path))
        elif self.onnx:            
            print('not get-units-2')
            units = get_onnx_units(self.hbt_model, wav_path).squeeze(0)
        elif hparams['use_cn_hubert'] or self.hubert_mode == 'cn_hubert':
            print('not get-units-3')
            units = get_cn_hubert_units(self.hbt_model, wav_path, self.dev).cpu().numpy()[0]
        else:
            units = get_units(self.hbt_model, wav_path, torch.device('cuda:6'), max_wav_length).cpu().numpy()[0]
            # print(len(units))
            # units = get_units(self.hbt_model, wav_path, self.dev, max_wav_length).cpu().numpy()[0]
            # torch.device('cuda')
        return units  # [T,256]
