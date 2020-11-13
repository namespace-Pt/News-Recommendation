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

## Achieved
### /utils/preprocess.py
- MINDIterator
  - read and parse data in MIND datasets
  - return a generator which generates *batch_size* of training examples once
  
### /utils/utils.py
  - some useful function

### /models
  - NPA[23]

## TODO
- integrate MINDIterator to Datasets and Dataloader

## Insights
### Conv1d
- calculate *signal_length* $L_{out}$ after convolution:
  - consider a sequence of length $L_{in}$ of signal of *in_channel*, then the output sequence is from $d * (k-1) + 1 - p$ to $L_{in} + p$, then $L_{out}$ as the number of convolution calculation can be derived as $$L_{out} = \frac{L_{in} - d * (k-1) - 1 + 2*p}{s} + 1$$where $d$ denotes *dilation rate*, $k$ denotes *kernel_size*, $p$ denotes *padding(on both sides)* and $s$ denotes *stride*

### Layer Normalization
- *mean and variance* is calculated on each sample rather over the whole batch