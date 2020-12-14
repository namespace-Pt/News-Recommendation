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
import torch.optim as optim
from datetime import datetime
from torchtext.vocab import GloVe
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.MIND import MIND_iter,MIND_map
from utils.utils import getLoss,constructBasicDict,run_eval,run_train
from models.NPA import NPAModel

if __name__ == "__main__":
    hparams = {
        'mode':sys.argv[1],
        'name':'npa',
        'batch_size':100,
        'title_size':30,
        'his_size':50,   
        'npratio':4,
        'dropout_p':0.2,
        'filter_num':400,
        'embedding_dim':300,
        'user_dim':50,
        'preference_dim':200,
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:0',
        'attrs': ['title'],
        'epochs':int(sys.argv[2])
    }

    news_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/news.tsv'
    news_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/news.tsv'
    news_file_pair = (news_file_train,news_file_test)

    behavior_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/behaviors.tsv'
    behavior_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/behaviors.tsv'
    behavior_file_pair = (behavior_file_train,behavior_file_test)

    save_path = 'models/model_params/{}_{}_{}'.format(hparams['name'],hparams['mode'],hparams['epochs']) +'.model'

    if not os.path.exists('data/dictionaries/vocab_{}_{}.pkl'.format(hparams['mode'],'_'.join(hparams['attrs']))):
        constructBasicDict(news_file_pair,behavior_file_pair,hparams['mode'],hparams['attrs'])

    device = torch.device(hparams['device']) if torch.cuda.is_available() else torch.device("cpu")

    dataset_train = MIND_map(hparams=hparams,news_file=news_file_train,behaviors_file=behavior_file_train)
    dataset_test = MIND_iter(hparams=hparams,news_file=news_file_test,behaviors_file=behavior_file_test)

    vocab = dataset_train.vocab
    embedding = GloVe(dim=300,cache='.vector_cache')
    vocab.load_vectors(embedding)

    loader_train = DataLoader(dataset_train,batch_size=hparams['batch_size'],shuffle=True,pin_memory=True,num_workers=20,drop_last=True)
    loader_test = DataLoader(dataset_test,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)

    npaModel = NPAModel(vocab=vocab,hparams=hparams,uid2idx=dataset_train.uid2index).to(device)
    
    if sys.argv[3] == 'eval':
        npaModel.load_state_dict(torch.load(save_path))
        npaModel.eval()
        

    elif sys.argv[3] == 'train':
        npaModel.train()
        writer = SummaryWriter('data/tb/{}/{}/{}/'.format(hparams['name'], hparams['mode'], datetime.now().strftime("%Y%m%d-%H")))

    if npaModel.training:
        print("training...")
        loss_func = getLoss(npaModel)
        optimizer = optim.Adam(npaModel.parameters(),lr=0.001)
        npaModel = run_train(npaModel,loader_train,optimizer,loss_func,writer,epochs=hparams['epochs'], interval=10)
        torch.save(npaModel.state_dict(), save_path)
        print("save success!")

    print("evaluating...")
    npaModel.cdd_size = 1
    npaModel.eval()
    run_eval(npaModel,loader_test)