This repository is under Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

### This repository is still in construction, there might still be some inconsistencies and lack of description

# PhonHuBERT
This repository contains the code for the PhonHuBERT: an aligned phoneme sequence transcription tool for Singing Voice Synthesis dataset automatic annotation.
In this work, we use HuBERT model as an encoder to generate hidden-space vectors (which would have clustered singing information) as it was done in this GitHub repository [DiffSVCS](https://github.com/prophesier/diff-svc).
We then propose two models:
- PhonHuBERT1 which decodes the framed phoneme sequence directly with a BLSTM-based network.
- PhonHuBERT2 which decodes the phoneme-ordered list with one BLSTM-based network and the onset-ordered list with another BLSTM-based network, before combining both information to generate the framed phoneme sequence of the song.

### Environment Preparation
All the experiments have been done in a conda environment with the Python version 3.9.16. All the dependencies can be found in the requirements.txt file.

To install the dependencies, run `pip -r install requirements.txt`

### Data Preparation

To reproduce our results, you first need the opencpop dataset, which can be downloaded following this link [opencpop](https://cloud.tsinghua.edu.cn/d/2870f80cb2c04b298d29/).

After downloading the dataset, ensure the opencpop dataset is saved under `/data/opencpop`. You can then run the following command to prepare the data:

```bash
python pre_processing.py
```

### Training

After the Data Preparation, you can train the model with the [trainer.py](trainer.py) file.

```bash
CUDA_VISIBLE_DEVICES=1,2 python trainer.py
```

### Inference







