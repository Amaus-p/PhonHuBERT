This repository is under Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

# PhonHuBERT
This repository contains the code for PhonHuBERT model: an Aligned Phoneme Sequence Transcription tool for Singing Voice Synthesis dataset automatic annotation.
In this work, we use HuBERT model as an encoder to generate hidden-space vectors (which would have clustered singing information) as it was done in this GitHub repository [DiffSVCS](https://github.com/prophesier/diff-svc). Our model is based on a serie of BLSTM layers to decode the encoded information by HuBERT and on MSE loss for the training part.

The figure below is a summary PhonHuBERT model.

![PhonHuBERT](https://github.com/Amaus-p/PhonHuBERT/blob/main/PhonHuBERT_and_legend.png?raw=true)

### Environment Preparation
All the experiments have been done in a conda environment with the Python version 3.9.16. All the dependencies can be found in the requirements.txt file.

To install the dependencies, run `pip -r install requirements.txt`

### Data Preparation

To reproduce our results, you first need the OpenCPOP dataset, which can be downloaded following this link [opencpop](https://cloud.tsinghua.edu.cn/d/2870f80cb2c04b298d29/).

After downloading the dataset, ensure the OpenCPOP dataset is saved under `/data/opencpop`. You can then run the following command to prepare the data:

```bash
python pre_processing.py
```

### Training

After the data preparation, you can train the model with the file [trainer.py](trainer.py). The file [constant.py](./utils/constants.py) contains all the hyperparameters interesting for the training and evaluation of the model. You can adpat the variables there to the experiment you want to reproduce.

```bash
CUDA_VISIBLE_DEVICES=1,2 python trainer.py
```

### Evaluation

For the inference and evaluation of the model, you can use the file [infer.py](infer.py). The results will be written in the folder [./results](./results).

```bash
CUDA_VISIBLE_DEVICES=1,2 python infer.py
```

We evaluated PhonHuBERT on the Phoneme Error Rate (PER). Our results displayed in the figure below.

![Results PhonHuBERT](https://github.com/Amaus-p/PhonHuBERT/blob/main/results_phonhubert.png?raw=true)

### Results example

The results files are composed of the name of the transcribed audiofile, the hyperparameters used for the training and then the predictions of the model. 
In order to evaluate our model on the phonemes themselves, we cut the silences before and after the singing part to only evaluate the part where the singer really produces sounds. 
For the results, the 3 values which have the higher predictions probability in the predicted probability vector v. For a frame t, if v1 is the phoneme with the highest probability, v2 the second highest, v3 the third highest and gt the ground truth value, the results we display are in the following format:

```txt
v3 v2 v1 | gt
```

Below are displayed the first lines of results for the file '2009.wav' with PhonHuBERT trained with the MSE loss.

```txt
e c t | SP
k a t | t
in k t | t
in k t | t
in k t | t
in k t | t
in k t | t
u h t | t
h u t | t
e ou u | u
ao ou u | u
ao ou u | u
uo ou u | u
uo ou u | u
r ou u | u
e r u | u
e r u | u
e u r | u
u e r | u
uan an r | r
an en r | r
en an r | r
en an r | an
r en an | an
r en an | an
r en an | an
r en an | an
a en an | an
e en an | an
en f an | an
AP an f | f
ie h f | f
ai ie f | f
ua k f | f
k t f | f
a t f | f
ao a f | f
an f a | a
iao an a | a
iao e a | a
eng e a | a
ie e a | a
y e a | a
y e a | a
y e a | a
ang e a | a
ang e a | a
y ang a | a
y ang a | a
iao ang a | a
iao ang a | a
ao a x | x
ao a x | x
ing ao x | x
ao ing x | x
ao ing x | x
ao ing x | x
ao ia x | x
ia n x | x
m n x | x
k h x | x
j h x | x
j ian x | x
iang x ian | ia
in iang ian | ia
j ie ian | ia
AP ie ian | ia
ang ie ian | ia
ang ie ian | ia
ang ie ian | ia
ang ie ian | ia
uang ie ian | ia
uang ie ian | ia
y ie ian | ia
y ie ian | ia
y ie ian | ia
y ie ian | ia
y ie ian | ia
ing ie ian | ia
ing ie ian | ia
ing ie ian | ia
AP ie ian | ia
j zh ian | ia
ian j zh | zh
ch j zh | zh
ch j zh | zh
ch j zh | zh
ch j zh | zh
zh ia e | zh
```

If you found this work helpful please consider citing our paper:

@InProceedings{10.1007/978-981-97-4399-5_12,
author="Prat, Amaury
and Yang, Runxuan
and Hu, Xiaolin",
editor="Le, Xinyi
and Zhang, Zhijun",
title="PhonHuBERT: A Phoneme Transcription Tool for Song Datasets",
booktitle="Advances in Neural Networks -- ISNN 2024",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="123--132",
abstract="In recent years, deep learning has gradually replaced traditional mathematical inference-based architectures, such as Hidden Markov Chains in Singing Voice Synthesis (SVS) systems, which led to an increase in the demand for accurately labeled data. In response to this need, this work introduces an Aligned Phoneme Sequence Transcription (APST) model for automatic song datasets annotation, called PhonHuBERT. This model uses HuBERT - a pre-trained self-supervised model for voice features classification - as an encoder, combined with Bidirectional Long-Short Term Memory (BLSTM) networks as a decoder.",
isbn="978-981-97-4399-5"
}






