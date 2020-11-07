# Code
## Requirements
```shell
python == 3.7.9
torch == 1.6.0
cuda == 10.1
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

## Instruction
```shell
run main.ipynb
```

## Question
- suppose tensor x of [1 * b * c], which is derived from MLP, then I copy the tensor along dim=0 for further computation, where I got tensor of [a * b * c], finally when training will the gradient bp be affected? 

## Insight
- considering negtive sampling from the current impression 