'''
Author: Pt
Date: 2020-11-10 00:14:02
LastEditTime: 2020-11-21 01:25:44
Description: 
'''
import os
import sys
os.chdir('/home/peitian_zhang/Codes/NR')
sys.path.append('/home/peitian_zhang/Codes/NR')

import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchtext.vocab import FastText
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.MIND import MIND_iter,MIND_map
from utils.utils import getLoss,getLabel,constructBasicDict,run_eval,run_train
from models.NPA import NPAModel

if __name__ == "__main__":
    hparams = {
        'mode':sys.argv[1],
        'batch_size':256,
        'title_size':30,
        'his_size':50,   
        'npratio':4,
        'dropout_p':0.2,
        'filter_num':400,
        'embedding_dim':300,
        'user_dim':50,
        'preference_dim':200,
        'metrics':'group_auc,ndcg@4,mean_mrr',
        'gpu':'cuda:0',
        'attrs': ['title'],
        'epochs':int(sys.argv[2])
    }

    news_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/news.tsv'
    news_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/news.tsv'

    behavior_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/behaviors.tsv'
    behavior_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/behaviors.tsv'

    save_path = '/home/peitian_zhang/Codes/NR/models/model_param/NPA_'+ hparams['mode'] +'.model'

    if not os.path.exists('data/dictionaries/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'train','_'.join(hparams['attrs']))):
        os.chdir('/home/peitian_zhang/Codes/NR/')
        constructBasicDict(news_file_train,behavior_file_train,hparams['mode'],'train',hparams['attrs'])

    if not os.path.exists('data/dictionaries/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'test','_'.join(hparams['attrs']))):
        os.chdir('/home/peitian_zhang/Codes/NR/')
        constructBasicDict(news_file_test,behavior_file_test,hparams['mode'],'test',hparams['attrs'])

    device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device("cpu")

    dataset_train = MIND_map(hparams=hparams,mode='train',news_file=news_file_train,behaviors_file=behavior_file_train)
    dataset_test = MIND_iter(hparams=hparams,mode='test',news_file=news_file_test,behaviors_file=behavior_file_test)

    vocab_train = dataset_train.vocab
    embedding = FastText('simple',cache='.vector_cache')
    vocab_train.load_vectors(embedding)

    vocab_test = dataset_test.vocab
    vocab_test.load_vectors(embedding)

    loader_train = DataLoader(dataset_train,batch_size=hparams['batch_size'],shuffle=True,pin_memory=True,num_workers=8,drop_last=True)
    loader_test = DataLoader(dataset_test,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)
    
    writer = SummaryWriter('data/tb/npa/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    try:
        if sys.argv[3] == 'eval':
            npaModel = NPAModel(vocab=vocab_train,hparams=hparams).to(device)
            npaModel.load_state_dict(torch.load(save_path))
            npaModel.eval()
        elif sys.argv[3] == 'train':
            npaModel = NPAModel(vocab=vocab_train,hparams=hparams).to(device)
            npaModel.train()
    
    except IndexError:
        npaModel = NPAModel(vocab=vocab_train,hparams=hparams).to(device)
        npaModel.train()

    if npaModel.training:
        print("training...")
        loss_func = getLoss(npaModel)
        optimizer = optim.Adam(npaModel.parameters(),lr=0.001)
        npaModel = run_train(npaModel,loader_train,optimizer,loss_func, writer,epochs=hparams['epochs'], interval=10)
  
    print("evaluating...")
    npaModel.eval()
    npaModel.vocab = vocab_test
    npaModel.npratio = -1

    print(run_eval(npaModel,loader_test))

    npaModel.npratio = 4
    torch.save(npaModel.state_dict(), save_path)