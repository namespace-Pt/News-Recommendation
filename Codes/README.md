# Code
## Requirements
```shell
python == 3.7.9
torch == 1.6.0
cuda == 10.1
```

## Instruction
```shell
run [model_name].ipynb
```

## File Structure
### `/utils`: data loader and utility functions
- `preprocess.py`
  - MINDIterator
    - read and parse data in MIND datasets
    - return a generator which generates *batch_size* of training examples once
      - if `npratio > 0`
        - generates positive candidates only
        - negtive sampling enabled
      - else
        - generates both positive and negtive candidates
        - negtive sampling disabled
 
- `utils.py`
  - some useful functions
    - construct dictionary
    - wrap training and testing/evaluating

### `/models`: achieved models
  - [NPA[23]](NPA.ipynb)
  - [FIM[29]](FIM.ipynb)

### `/data`: basic dictionaries
  - vocab
  - nid2idx
  - uid2idx

### `/tips`: insight of some api
  - torch_tips.ipynb
    - useful api, confusing api

## TODO
- [ ] integrate MINDIterator to Datasets and Dataloader
  - motivation: split data with more flexibility
  - **falied**, need to explore datasets which returns several tensors
- [ ] understand *permute*
- [x] construct `nid2idx` and `uid2idx` according to both training iterator and testing iterator
- [ ] k-fold validation, concepts and implementation
- [ ] analyze MIND dataset, calculate average user history length

## Insights
### Convolution
- calculate *signal_length* $L_{out}$ after convolution:
  - consider a sequence of length $L_{in}$ of signal of *in_channel*, then the output sequence is from $d * (k-1) + 1 - p$ to $L_{in} + p$, then $L_{out}$ as the number of convolution calculations can be derived as $$L_{out} = \frac{L_{in} - d * (k-1) - 1 + 2*p}{s} + 1$$where $d$ denotes *dilation rate*, $k$ denotes *kernel_size*, $p$ denotes *padding(on both sides)* and $s$ denotes *stride*

### Layer Normalization
- *mean and variance* is calculated on each sample rather over the whole batch

### [Pytorch Manipulation](tips/torch_tips.ipynb)