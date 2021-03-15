# Code
## Requirements
```shell
python == 3.8.5
torch == 1.7.1
torchtext == 0.8.1
cuda == 10.1
pandas
tqdm
scikit-learn
```
## Rank
[Leaderboard](https://competitions.codalab.org/competitions/24122#results), User Name: **Pt**
## Dataset
download MIND dataset [HERE](https://msnews.github.io/)
### Simple Analysis on MIND
see [Preprocess.ipynb](manual/Preprocess.ipynb)

## Instruction
- **check out [Preprocess.ipynb](manual/Preprocess.ipynb) first to get familiar with datasets**
- you can customize your **MIND directory** by calling `prepare(hparams, path='\your\path\MIND')`
  
- you can run the specific notebook to train and test the model
  ```shell
  run manual/[model_name].ipynb
  ```

- you can alse run **python scripts** in terminal, type in `--help/-h` to get detail explanation of arguments
  ```shell
  usage: [script_name].py [-h] -s {demo,small,large} [-m {train,dev,test,tune}] [-e EPOCHS] [-bs BATCH_SIZE] [-ts TITLE_SIZE] [-hs HIS_SIZE] [-c {0,1}] [-se SAVE_EACH_EPOCH] [-ss SAVE_STEP]
                [-te TRAIN_EMBEDDING] [-lr LEARNING_RATE] [-np NPRATIO] [-mc METRICS] [-k K] [--select {pipeline1,pipeline2,unified,gating}] [--integrate {gate,harmony}]
                [-hn HEAD_NUM] [-vd VALUE_DIM] [-qd QUERY_DIM] [-at ATTRS] [-v] [-nid]
  ```
  - **e.g. train FIM model on MINDlarge for 2 epochs. At the end of each step, save the model, meanwhile, save model every 2000 steps**
    ```shell
    cd Codes/
    python scripts/fim.py --scale large --mode train --epoch 2 --save_each_epoch --save_step 2000
    ```
  - **e.g. test FIM model which were trained on `MINDlarge` and saved at the end of 4000 steps**
    ```shell
    cd Codes/
    python scripts/fim.py -s large -m test --save_step 4000
    ```
- **ATTENTION! default path of model parameters is** `models/model_params/[model_name]`, since I didn't upload this folder, you need to create one.

- **ATTENTION! codes under `deprecated/` will no longer be maintained**
   

## File Structure
### `/data`: basic dictionaries
  - dictionary which maps News ID to increasing integer, training set and testing set are separate because when constructing MIND iterator, the newsID should be mapped to continuous number starting from 1
    - `nid2idx_[data_mode]_train.json`
    - `nid2idx_[data_mode]_test.json`
  - dictionary which maps News ID to increasing integer, training set and testing set are unified because user may appear in both training and testing phases which are namely *long-tail users*. However, some users may only appear in testing set, which fomulates *cold start problem*.
    - `uid2idx_[data_mode].json`
  - vocabulary which maps word wokens to increasing integer (**instance of torchtest.vocab**) , which can be applied with pre-trained word embeddings.
    - `vocab_[data_mode].pkl`
  - `/tb`
    - `/[model_name]`
      - log file for `Tensorboard`

### `/manual`: jupyter notebooks for training and testing models
  - [Preprocess.ipynb](manual/Preprocess.ipynb)
    - viewing data
  - [torch_tips.ipynb](manual/torch_tips.ipynb)
    - manipulation over `PyTorch`

### `/models`: reproduced models and my models
  - NPA
    - [[23] Npa Neural news recommendation with personalized attention](https://dl.acm.org/doi/abs/10.1145/3292500.3330665)
  - FIM
    - [[29] Fine-grained Interest Matching for Neural News Recommendation](https://www.aclweb.org/anthology/2020.acl-main.77.pdf)
  - NRMS
    - [[22] Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671.pdf)
  - KNRM
    - [[49] End-to-End Neural Ad-hoc Ranking with Kernel Pooling](https://dl.acm.org/doi/pdf/10.1145/3077136.3080809)  
  - Soft Top-k Operator 
    - [[51] Differentiable Top-K Operator with Optimal Transport](https://arxiv.org/pdf/2002.06504.pdf)
    - *copy code from paper*

### `/scripts`: python scripts of training/testing

### `/utils`: data loader and utility functions
- `MIND.py`
  - MIND
    - iterable dataset for MIND
    - for each behavior log, either yielding positive click behaviors only (training) or yielding every behavior (validating)
  - MIND_test
    - `MINDlarge_test` contain no labels, so is picking out, the function is the same as MIND
  - MIND_news
    - map style dataset for MIND/news
    - sequentially returning each news token for learning news embedding/representation in pipelines

- `utils.py`
  - some helper functions
    - construct dictionary
    - wrap training and testing/evaluating

## Experiment
**the model is run on `MINDlarge` if not specified**
|model|AUC|MRR|NDCG@5|NDCG@10|benchmark-achieve-at|
|:-:|:-:|:-:|:-:|:-:|:-:|
|NPA|$0.6705$|$0.3147$|$0.3492$|$0.4118$|`epoch=5`|
|FIM|$0.678$|$0.3292$|$0.3655$|$0.4266$|`step=10000`|
|NRMS|$0.6618$|$0.3179$|$0.3444$|$0.4108$|`epoch=6`|
|ITR-CNN-CNN|$0.647$|$0.3022$|$0.3289$|$0.3957$|`epoch=1`|
|ITR-MHA-MHA|$0.6201$|||
|SFI-pipeline||
|SFI-unified|$0.6782$|$0.3237$|$0.3598$|$0.4237$|
|SFI-gating|$\mathbf{0.6853}$|$\mathbf{0.3303}$|$\mathbf{0.3663}$|$\mathbf{0.4294}$|`epoch=1, step=18000`|
|baseline-MHA-CNN|$0.6239$|$0.2796$|$0.3067$|$0.3698$|`epoch=8` run on `MINDsmall`|
|baseline-MHA-MHA|$0.6395$|$0.2934$|$0.3203$|$0.385$|`epoch=8` run on `MINDsmall`|

## TODO
- [x] BERT
- [ ] t-test
- [x] inference time comparison
- [x] pipeline performance comparison
- [ ] reproduce Hi-Fi Ark
- [x] sfi frame