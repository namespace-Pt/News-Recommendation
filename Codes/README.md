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

## Question
- suppose tensor x of [1 * b * c], which is derived from MLP, then I copy the tensor along dim=0 for further computation, thus I got tensor of [a * b * c], finally when training will the gradient backpropagating be affected? 
- batch is faster than loop?
- how to compute auc, what is threshold in sklearn.metrics.roc_curve?

## Insight
- considering negtive sampling from the current impression 