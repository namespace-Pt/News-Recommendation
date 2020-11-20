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
```shell
run [model_name].ipynb
```
you can alse run **python scripts** in terminal directly
```shell
cd Codes/
python scripts/[model_name].py
```
*remember to customize your paths and hyper parameters*

## File Structure
### `/data`: basic dictionaries
  - vocab
  - nid2idx
  - uid2idx

### `/models`: achieved models
  - [NPA[23]](NPA.ipynb)
  - [FIM[29]](FIM.ipynb)

### `/scripts`: python scripts of models
  - same as `[model_name].ipynb`, you can run it in terminal  

### `/tips`: insight of some api
  - torch_tips.ipynb
    - useful api, confusing api

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