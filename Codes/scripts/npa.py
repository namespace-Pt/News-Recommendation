'''
Author: Pt
Date: 2020-11-20 01:10:32
LastEditTime: 2020-11-20 10:43:23
Description: 
'''
import os
import sys
os.chdir('D:\\repositories\\News-Recommendation\\Codes')
sys.path.append('D:\\repositories\\News-Recommendation\\Codes')

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import FastText
from torch.utils.data import DataLoader
from utils.MIND import MIND_iter,MIND_map
from utils.utils import getLoss,getLabel,constructBasicDict,run_eval,run_train
from models.NPA import NPAModel

if __name__ == "__main__":
    hparams = {
        'mode':'demo',
        'batch_size':5,#100,
        'title_size':30,
        'his_size':20,
        'dropout_p':0.2,
        'filter_num':400,
        'embedding_dim':300,
        'user_dim':50,
        'preference_dim':200,
        'metrics':'group_auc,ndcg@4,mean_mrr',
        'gpu':'cuda:0',
        'attrs': ['title']
    }

    # customize your path here

    news_file_train = 'D:/Data/NR_data/dev/news_train.tsv'
    news_file_test = 'D:/Data/NR_data/dev/news_test.tsv'
    behavior_file_train = 'D:/Data/NR_data/dev/behaviors_train.tsv'
    behavior_file_test = 'D:/Data/NR_data/dev/behaviors_test.tsv'
    save_path = 'models/model_param/NPA_'+ hparams['mode'] +'.model'

    # if user2id,word2id,news2id haven't been constructed
    if not os.path.exists('data/nid2idx_{}_{}.json'.format(hparams['mode'],'train')):
        constructBasicDict(news_file_train,behavior_file_train,hparams['mode'],'train',hparams['attrs'])

    if not os.path.exists('data/nid2idx_{}_{}.json'.format(hparams['mode'],'test')):
        constructBasicDict(news_file_test,behavior_file_test,hparams['mode'],'test',hparams['attrs'])

    device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device("cpu")

    dataset_train = MIND_map(hparams=hparams,mode='train',npratio=4,news_file=news_file_train,behaviors_file=behavior_file_train)

    dataset_test = MIND_iter(hparams=hparams,mode='test',news_file=news_file_test,behaviors_file=behavior_file_test)

    vocab_train = dataset_train.vocab
    embedding = FastText('simple',cache='.vector_cache')
    vocab_train.load_vectors(embedding)

    vocab_test = dataset_test.vocab
    vocab_test.load_vectors(embedding)

    loader_train = DataLoader(dataset_train,batch_size=hparams['batch_size'],shuffle=True,pin_memory=True,num_workers=3)
    loader_test = DataLoader(dataset_test,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)

    # you can load my model or train yours
    if os.path.exists(save_path):
        npaModel = NPAModel(vocab=vocab_train,hparams=hparams,npratio=4).to(device)
        npaModel.load_state_dict(torch.load(save_path))
        npaModel.eval()

    else:
        npaModel = NPAModel(vocab=vocab_train,hparams=hparams,npratio=4).to(device)
        npaModel.train()

    if npaModel.training:
        print("training...")
        loss_func = getLoss(npaModel)
        optimizer = optim.Adam(npaModel.parameters(),lr=0.001)
        npaModel = run_train(npaModel,loader_train,optimizer,loss_func, epochs=1, interval=5)

    print("evaluating...")
    npaModel.eval()
    npaModel.vocab = vocab_test
    npaModel.npratio = -1

    run_eval(npaModel,loader_test)