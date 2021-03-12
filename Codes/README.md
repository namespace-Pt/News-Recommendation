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
### Simple Analysis
see [Preprocess.ipynb](manual/Preprocess.ipynb)
## Instruction
- **check out [Preprocess.ipynb](manual/Preprocess.ipynb) first to get familiar with datasets**
- **you can customize your dataset path in two ways:**
  - explicitly pass `path` parameter your own **top directory of `MINDxxxx`** when calling `prepare()`
  
- you can run the specific notebook to train and test the model
  ```shell
  run manual/[model_name].ipynb
  ```

- you can alse run **python scripts** in terminal, type in `--help/-h` to get detail explanation of arguments
  ```shell
  usage: [script_name].py [-h] -s {demo,small,large} [-m {train,dev,test}] [-e EPOCHS] [-bs BATCH_SIZE] [-ts TITLE_SIZE] [-hs HIS_SIZE] [-c {0,1}] [-se]
              [-ss SAVE_STEP] [-te] [-np NPRATIO] [-mc METRICS] [-k K] [--select {pipeline,unified,gating}] [--integrate {gate,harmony}]
              [-hn HEAD_NUM] [-vd VALUE_DIM] [-qd QUERY_DIM] [-v] [-nid]
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
- **ATTENTION! default path of model parameters is**
  ```
  models/model_params
  ``` 

## File Structure
### `/data`: basic dictionaries
  - dictionary mapping News ID to increasing integer, training set and testing set are separate because when constructing MIND iterator, the newsID should be mapped to continuous number starting from 1
    - `nid2idx_[data_mode]_train.json`
    - `nid2idx_[data_mode]_test.json`
  - dictionary mapping News ID to increasing integer, training set and testing set are unified because user may appear in both training and testing phases which are namely *long-tail users*. However, some users may only appear in testing set, which fomulates *cold start problem*.
    - `uid2idx_[data_mode].json`
  - vocabulary mapping word wokens to increasing integer (**instance of torchtest.vocab**) , which can be applied with pre-trained word embeddings.
    - `vocab_[data_mode].pkl`
  - `/tb`
    - `/[model_name]`
      - log file for `Tensorboard`

### `/manual`: jupyter notebooks for training and testing models
  - [NPA.ipynb](manual/NPA.ipynb)
  - [FIM.ipynb](manual/FIM.ipynb)

  - [Preprocess.ipynb](manual/Preprocess.ipynb)
    - viewing data
  - [torch_tips.ipynb](manual/torch_tips.ipynb)
    - manipulation over `PyTorch`

### `/models`: reproduced models
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

### `/scripts`: python scripts of models
  - you can run models in `shell`

### `/utils`: data loader and utility functions
- `MIND.py`
  - MIND
    - iterable dataset for MIND
    - when **training**
  - MIND_iter
    - iterator style dataset
    - return dictionary of one candidate news
      - negtive sampling disabled, **point-wise**, intended for evaluating

- `utils.py`
  - some helper functions
    - construct dictionary
    - wrap training and testing/evaluating
## Experiment
### Hyper Parameter Settings
- `title-size=20`
  - the length of news title, pad 0 if less, strip off if more
  - none of category, subcategory and absract is involved
- `his-size=50`
  - the length of user history, pad 0 if less, strip off if more
  - in `baseline`, `his-size=100`

### Performance
**the model is run on `MINDlarge` if not specified**
|model|AUC|MRR|NDCG@5|NDCG@10|benchmark-achieve-at|
|:-:|:-:|:-:|:-:|:-:|:-:|
|NPA|$0.6705$|$0.3147$|$0.3492$|$0.4118$|`epoch=5`|
|FIM|$0.678$|$0.3292$|$0.3655$|$0.4266$|`step=10000`|
|NRMS|$0.6618$|$0.3179$|$0.3444$|$0.4108$|`epoch=6`|
|ITR-CNN-CNN|$0.647$|$0.3022$|$0.3289$|$0.3957$|`epoch=1`|
|ITR-MHA-MHA|$0.6201$|||
|SFI-pipeline||
|SFI-unified||
|SFI-gating|$\mathbf{0.6853}$|$\mathbf{0.3303}$|$\mathbf{0.3663}$|$\mathbf{0.4294}$|`epoch=1, step=18000`|
|baseline-MHA-CNN|$0.6239$|$0.2796$|$0.3067$|$0.3698$|`epoch=8` run on `MINDsmall`|
|baseline-MHA-MHA|$0.6395$|$0.2934$|$0.3203$|$0.385$|`epoch=8` run on `MINDsmall`|

## TODO
- [ ] BERT
- [ ] t-test
- [ ] inference time comparison
- [ ] pipeline performance comparison
- [ ] logging
- [ ] reproduce Hi-Fi Ark
- [ ] parallel evaluating/training
  