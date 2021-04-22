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
[Leaderboard](https://msnews.github.io/#leaderboard), User Name: **Pt**
## Dataset
download MIND dataset [HERE](https://msnews.github.io/)
### Simple Analysis on MIND
see [Preprocess.ipynb](manual/Preprocess.ipynb)
```
 avg_title_length:10.67731736385395
 avg_abstract_length:36.4448570331045
 avg_his_length:32.99787212887438
 avg_impr_length:37.40116394684935
 cnt_his_lg_50:447829
 cnt_his_eq_0:46065
 cnt_imp_multi:567571
```

## Instruction
- **check out [Preprocess.ipynb](manual/Preprocess.ipynb) first to get familiar with datasets**
- you can customize your **MIND directory** by calling `prepare(hparams, path='\your\path\MIND')`

- you can run the specific notebook to train and test the model
  ```shell
  run manual/[model_name].ipynb
  ```

- you can alse run **python scripts** in terminal, type in `--help/-h` to get detail explanation of arguments
  ```shell
  usage: [model].py [-h] -s {demo,small,large,whole} [-m {train,dev,test,tune,encode}] [-e EPOCHS] [-bs BATCH_SIZE] [-ts TITLE_SIZE] [--abs_size ABS_SIZE]
              [-hs HIS_SIZE] [--device {0,1,cpu}] [--save_step SAVE_STEP] [--val_freq VAL_FREQ] [-ck CHECKPOINT] [-lr LEARNING_RATE]
              [--schedule SCHEDULE] [--npratio NPRATIO] [-mc METRICS] [--topk K] [--contra_num CONTRA_NUM]
              [--select {pipeline1,pipeline2,unified,gating}] [--integrate {gate,harmony}] [--encoder ENCODER] [--interactor INTERACTOR] [--dynamic]
              [--bert {bert-base-uncased,albert-base-v2}] [--level LEVEL] [--pipeline PIPELINE] [-hn HEAD_NUM] [-vd VALUE_DIM] [-qd QUERY_DIM]
              [--attrs ATTRS] [--validate]
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
- **default path of model parameters is** `data/model_params/[model_name]`, since I didn't upload this folder, you need to create one.

- **ATTENTION! codes under `deprecated/` will no longer be maintained**


## File Structure
### `/data`: basic dictionaries
  - dictionary which maps News ID to increasing integer, training set and testing set are separate because when constructing MIND iterator. **News index starts from 1.**
    - `nid2idx_[data_mode]_train.json`
    - `nid2idx_[data_mode]_test.json`
  - dictionary which maps News ID to increasing integer, training set and testing set are unified because user may appear in both training and testing phases. However, some users may only appear in testing set, which fomulates *cold start problem*. **User index starts from 1.**
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
    - map style dataset for MIND
    - for each behavior log, either returning positive click behaviors only (training) or returning every behavior (validating/testing)
    ```python
    {
        "impression_index": ,
        "user_index": ,
        'cdd_id': ,
        "candidate_title": ,
        "candidate_title_pad": ,
        "candidate_abs": ,
        "candidate_abs_pad": ,
        "candidate_vert": ,
        "candidate_subvert": ,
        'his_id': ,
        "clicked_title": ,
        "clicked_title_pad": ,
        "clicked_abs": ,
        "clicked_abs_pad": ,
        "clicked_vert": ,
        "clicked_subvert": ,
        "his_mask":
    }
    ```

  - MIND_news
    - map style dataset for MIND/news
    - sequentially returning each news token for learning news embedding/representation in pipelines

- `utils.py`
  - some helper functions
    - construct dictionary
    - wrap training and testing/evaluating

## TODO
- [x] BERT
- [ ] t-test
- [x] inference time comparison
- [x] pipeline performance comparison
- [ ] reproduce Hi-Fi Ark
- [x] sfi frame
- [x] upgrade MIND iterable dataset to MIND map dataset
- [ ] subspace disentanglement
- [ ] abstract out the model properties and hyper parameter settings
- [ ] check original SFI's attention weight
- [ ] contrasive learning optimization
- [ ] lstm on embedding before/after dilated cnn
- [ ] position embedding
- [ ] FIM comparison for topk effectiveness
- [ ] different threshold
- [ ] delicate selection project compared with raw attention weight
- [ ] base_encoder

## Mem
- SFI-dynamic threshold=0.5 bad
- not sufficient enhancement with LSTM applied in encoder