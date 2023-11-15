This repository is under Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

# PhonHuBERT
This repository contains the code for the PhonHuBERT: an aligned phoneme sequence transcription tool for Singing Voice Synthesis dataset automatic annotation.
In this work, we HuBERT model as a decoder as it was done in this and this GitHub repository [DiffSVCS](https://github.com/prophesier/diff-svc).

### Environment Preparation
All the experiments have been made in a conda environment with the Python version 3.9.16. All the dependencies can be found in the requirements.txt file.

To install the dependencies, run `pip -r install requirements.txt`

### Data Preparation

To reproduce our results, you first need the opencpop dataset, which can be downloaded following this link [opencpop](https://cloud.tsinghua.edu.cn/d/2870f80cb2c04b298d29/)


### Training

After the Data Preparation, you can train the model with the [trainer.py](trainer.py) file.

```bash
CUDA_VISIBLE_DEVICES=1,2 python trainer.py
```

### Inference





