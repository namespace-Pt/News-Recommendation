'''
Author: Pt
Date: 2020-11-16 23:54:45
LastEditTime: 2020-11-21 01:25:01
Description: 
'''
import os
import sys
os.chdir('/home/peitian_zhang/Codes/NR')
sys.path.append('/home/peitian_zhang/Codes/NR')

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchtext.vocab import GloVe
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.MIND import MIND_iter,MIND_map
from utils.utils import getLoss,getLabel,constructBasicDict,run_eval,run_train
from models.FIM import FIMModel

if __name__ == "__main__":
    hparams = {
        'mode':sys.argv[1],
        'batch_size':100,
        'title_size':18,
        'his_size':50,
        'kernel_size':3,
        'npratio':4,
        'dilation_level':3,
        'filter_num':150,
        'embedding_dim':300,
        'metrics':'group_auc,ndcg@4,mean_mrr',
        'gpu':'cuda:0',
        'attrs': ['title','category','subcategory'],
        'epochs':int(sys.argv[2])
    }

    news_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/news.tsv'
    news_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/news.tsv'

    behavior_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/behaviors.tsv'
    behavior_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/behaviors.tsv'

    save_path = '/home/peitian_zhang/Codes/NR/models/model_param/FIM_'+ hparams['mode'] +'.model'

    if not os.path.exists('data/dictionaries/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'train','_'.join(hparams['attrs']))):
        constructBasicDict(news_file_train,behavior_file_train,hparams['mode'],'train',hparams['attrs'])

    if not os.path.exists('data/dictionaries/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'test','_'.join(hparams['attrs']))):
        constructBasicDict(news_file_test,behavior_file_test,hparams['mode'],'test',hparams['attrs'])

    device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device("cpu")

    dataset_train = MIND_map(hparams=hparams,mode='train',news_file=news_file_train,behaviors_file=behavior_file_train)
    dataset_test = MIND_iter(hparams=hparams,mode='test',news_file=news_file_test,behaviors_file=behavior_file_test)

    vocab_train = dataset_train.vocab
    embedding = GloVe(dim=300,cache='.vector_cache')
    vocab_train.load_vectors(embedding)

    vocab_test = dataset_test.vocab
    vocab_test.load_vectors(embedding)

    loader_train = DataLoader(dataset_train,batch_size=hparams['batch_size'],shuffle=True,pin_memory=True,num_workers=8,drop_last=True)
    loader_test = DataLoader(dataset_test,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)

    writer = SummaryWriter('data/tb/fim/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # you can load my model or train yours
    try:
        if sys.argv[3] == 'eval':
            fimModel = FIMModel(vocab=vocab_train,hparams=hparams).to(device)
            fimModel.load_state_dict(torch.load(save_path))
            fimModel.eval()
        elif sys.argv[3] == 'train':
            npaModel = FIMModel(vocab=vocab_train,hparams=hparams).to(device)
            npaModel.train()
    
    except IndexError:
        fimModel = FIMModel(vocab=vocab_train,hparams=hparams).to(device)
        fimModel.train()

    if fimModel.training:
        print("training...")
        loss_func = getLoss(fimModel)
        optimizer = optim.Adam(fimModel.parameters(),lr=0.001)
        fimModel = run_train(fimModel,loader_train,optimizer,loss_func,writer, epochs=hparams['epochs'], interval=100)

    print("evaluating...")
    fimModel.eval()
    fimModel.vocab = vocab_test
    fimModel.npratio = -1

    print(run_eval(fimModel,loader_test))

    fimModel.npratio = 4
    torch.save(fimModel.state_dict(), save_path)