# Code
## Requirements
```shell
python == 3.7.9
torch == 1.6.0
cuda == 10.1
```

## Dataset
download MIND dataset [**here**](https://msnews.github.io/), and customize data path in `[model_name].ipynb` and `scripts/[model_name].py`

## Instruction
*both need customing paths and hyper parameters in advance*
- you can run the specific notebook to train and test the model
  ```shell
  run manual/[model_name].ipynb
  ```

- you can alse run **python scripts** in terminal provided `data_scale`, `epochs`, and `mode` parameters. **eg:**
  ```shell
  cd Codes/
  python server_scripts/[model_name].py large 10 train
  ```

## File Structure
### `/data`: basic dictionaries
  - vocab
  - nid2idx
  - uid2idx
  - `/tb`
    - `/[model_name]`
      - log file for `Tensorboard`

### `/manual`: jupyter notebooks for training and testing models
  - [NPA.ipynb](manual/NPA.ipynb)
    - [[23] Npa Neural news recommendation with personalized attention](https://dl.acm.org/doi/abs/10.1145/3292500.3330665)
  - [FIM.ipynb](manual/FIM.ipynb)
    - [[29] Fine-grained Interest Matching for Neural News Recommendation](https://www.aclweb.org/anthology/2020.acl-main.77.pdf)

  - [Preprocess.ipynb](manual/Preprocess.ipynb)
    - viewing data
  - [torch-tips.ipynb](manual/torch-tips.ipynb)
    - manipulation over *pytorch*

### `/models`: achieved models
  - NPA
  - FIM
  - [[51] Differentiable Top-K Operator with Optimal Transport](https://arxiv.org/pdf/2002.06504.pdf)
  - [[22] Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671.pdf),
  - [[52] Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

### `/server_scripts`: python scripts of models
  - you can run models in *Ubuntu* server

### `/utils`: data loader and utility functions
- `MIND.py`
  - MIND_map
    - map style dataset
    - return dictionary of one behavior log
      - negtive sampling enabled
  - MIND_iter
    - iterator style dataset
    - return dictionary of one candidate news
      - negtive sampling disabled, **point-wise**, intended for evaluating

- `utils.py`
  - some useful functions
    - construct dictionary
    - wrap training and testing/evaluating

## TODO
- [x] integrate MINDIterator to Datasets and Dataloader
  - motivation: split data with more flexibility, enable distribution
- [ ] understand *permute*
- [x] construct `nid2idx` and `uid2idx` according to both training iterator and testing iterator
- [ ] analyze MIND dataset, calculate average user history length

## Insights
### Convolution
- calculate *signal_length* $L_{out}$ after convolution:
  - consider a sequence of length $L_{in}$ of signal of *in_channel*, then the output sequence is from $d * (k-1) + 1 - p$ to $L_{in} + p$, then $L_{out}$ as the number of convolution calculations can be derived as $$L_{out} = \frac{L_{in} - d * (k-1) - 1 + 2*p}{s} + 1$$where $d$ denotes *dilation rate*, $k$ denotes *kernel_size*, $p$ denotes *padding(on both sides)* and $s$ denotes *stride*

### Layer Normalization
- *mean and variance* is calculated on each sample rather over the whole batch

### [Pytorch Manipulation](tips/torch_tips.ipynb)