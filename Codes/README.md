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