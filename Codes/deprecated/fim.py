'''
Author: Pt
Date: 2020-11-16 23:54:45
LastEditTime: 2020-11-18 10:18:28
Description: 
'''
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import FastText
from utils.preprocess import MINDIterator
from utils.utils import getVocab,getLoss,getLabel,constructBasicDict,run_eval,run_train
from models.FIM import FIMModel

if __name__ == "__main__":
    hparams = {
        'npratio':4,
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

    if not os.path.exists('data/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'train','_'.join(hparams['attrs']))):
        os.chdir('/home/peitian_zhang/Codes/NR/')
        constructBasicDict(news_file_train,behavior_file_train,hparams['mode'],'train',hparams['attrs'])

    if not os.path.exists('data/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'test','_'.join(hparams['attrs']))):
        os.chdir('/home/peitian_zhang/Codes/NR/')
        constructBasicDict(news_file_test,behavior_file_test,hparams['mode'],'test',hparams['attrs'])

    device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device("cpu")

    iterator_train = MINDIterator(hparams=hparams,mode='train',news_file=news_file_train,behaviors_file=behavior_file_train)

    iterator_test = MINDIterator(hparams=hparams,mode='test',news_file=news_file_test,behaviors_file=behavior_file_test)

    vocab_train = iterator_train.vocab
    embedding = FastText('simple',cache='.vector_cache')
    vocab_train.load_vectors(embedding)

    vocab_test = iterator_test.vocab
    vocab_test.load_vectors(embedding)

    if os.path.exists(save_path):
        fimModel = FIMModel(vocab=vocab_train,hparams=hparams)
        fimModel.load_state_dict(torch.load(save_path))
        fimModel.to(device).eval()
    else:
        fimModel = FIMModel(vocab=vocab_train,hparams=hparams).to(device)
        fimModel.train()
    
    if fimModel.training:
        loss_func = getLoss(fimModel)
        optimizer = optim.Adam(fimModel.parameters(),lr=0.001)
        fimModel = run_train(fimModel,iterator_train,optimizer,loss_func, epochs=hparams['epochs'], interval=100)
    
    fimModel.eval()
    fimModel.vocab = vocab_test
    fimModel.npratio = -1
    iterator_test.npratio = -1

    print(run_eval(fimModel,iterator_test))

    fimModel.npratio = 4
    torch.save(fimModel.state_dict(), save_path)