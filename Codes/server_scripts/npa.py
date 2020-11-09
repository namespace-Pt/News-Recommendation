import sys
sys.path.append('..')

import os
import torch
import torch.optim as optim
from torchtext.vocab import FastText
from utils.preprocess import MINDIterator
from utils.utils import getVocab,getLoss,getLabel
from models.NPA import NPAModel
from utils.utils import run_eval

news_file = '/Data/NR_data/MINDsmall_train/news.tsv'
behavior_file = '/Data/NR_data/dev/behaviors_small.tsv'
hparams = {
    'mode':'small',
    'batch_size':100,
    'title_size':30,
    'his_size':50,   
    'npratio':4,     
    'dropout_p':0.2,
    'filter_num':400,
    'embedding_dim':300,
    'user_dim':50,
    'preference_dim':200,
    'metrics':'group_auc,ndcg@4,mean_mrr',
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

iterator = MINDIterator(hparams=hparams)
train = iterator.load_data_from_file(news_file,behavior_file)

# torchtext.Vocab.vocab object
vocab = iterator.word_dict
embedding = FastText('simple',cache='.vector_cache')
vocab.load_vectors(embedding)

npaModel = NPAModel(vocab=vocab,hparams=hparams)

# migrate the model to GPU
npaModel.to(device).train()

loss_func = getLoss(npaModel)
optimizer = optim.Adam(npaModel.parameters(),lr=0.0002)

num = 0
for epoch in range(10):
    train = iterator.load_data_from_file(news_file,behavior_file)

    for x in train:
        pred = npaModel(x)
        label = getLabel(npaModel,x)
        loss = loss_func(pred,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        num+=1
    
    print("epoch:{},loss:{}".format(epoch,loss))

npaModel.eval()

iterator.npratio = -1
test = iterator.load_data_from_file(news_file,behavior_file)

npaModel.npratio = -1
print(run_eval(npaModel,test))