'''
Author: Pt
Date: 2020-11-10 00:14:02
LastEditTime: 2020-11-18 10:18:38
Description: 
'''
import os
import sys
import torch
import torch.optim as optim
from torchtext.vocab import FastText
from utils.preprocess import MINDIterator
from utils.utils import getVocab,getLoss,getLabel,constructBasicDict,run_eval,run_train
from models.NPA import NPAModel

if __name__ == "__main__":
    hparams = {
        'npratio':4,
        'mode':sys.argv[1],
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
        'gpu':'cuda:0',
        'attrs': ['title'],
        'epochs':int(sys.argv[2])
    }

    news_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/news.tsv'
    news_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/news.tsv'

    behavior_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/behaviors.tsv'
    behavior_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/behaviors.tsv'

    save_path = '/home/peitian_zhang/Codes/NR/models/model_param/NPA_'+ hparams['mode'] +'.model'

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
        npaModel = NPAModel(vocab=vocab_train,hparams=hparams)
        npaModel.load_state_dict(torch.load(save_path))
        npaModel.to(device).eval()
    else:
        npaModel = NPAModel(vocab=vocab_train,hparams=hparams).to(device)
        npaModel.train()
    
    if npaModel.training:
        loss_func = getLoss(npaModel)
        optimizer = optim.Adam(npaModel.parameters(),lr=0.0002)
        npaModel = run_train(npaModel,iterator_train,optimizer,loss_func, epochs=hparams['epochs'], interval=100)
    
    npaModel.eval()
    npaModel.vocab = vocab_test
    npaModel.npratio = -1
    iterator_test.npratio = -1

    print(run_eval(npaModel,iterator_test))
    
    npaModel.npratio = 4
    torch.save(npaModel.state_dict(), save_path)