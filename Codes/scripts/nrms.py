import os
import sys
os.chdir('/home/peitian_zhang/Codes/NR')
sys.path.append('/home/peitian_zhang/Codes/NR')

import torch
import torch.optim as optim
from datetime import datetime
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.MIND import MIND_iter,MIND_map
from utils.utils import getLoss,constructBasicDict,run_eval,run_train
from models.NRMS import NRMSModel

if __name__ == "__main__":
    hparams = {
        'mode':sys.argv[1],
        'name':'nrms',
        'batch_size':64,
        'title_size':30,
        'his_size':50,
        'npratio':4,
        'dropout_p':0.2,
        'query_dim':200,
        'embedding_dim':300,
        'value_dim':16,
        'head_num':16,
        'kernel_num':11,
        'epochs':int(sys.argv[2]),
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:0',
        'attrs': ['title'],
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
    dataset_test = MIND_iter(hparams=hparams,news_file=news_file_test,behaviors_file=behavior_file_test, mode='test')

    vocab = dataset_train.vocab
    embedding = GloVe(dim=300,cache='.vector_cache')
    vocab.load_vectors(embedding)

    loader_train = DataLoader(dataset_train,batch_size=hparams['batch_size'],shuffle=True,pin_memory=True,num_workers=3,drop_last=True)
    loader_test = DataLoader(dataset_test,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)

    nrmsModel = NRMSModel(vocab=vocab,hparams=hparams).to(device)

    if sys.argv[3] == 'eval':
        nrmsModel.load_state_dict(torch.load(save_path))
        nrmsModel.eval()
        

    elif sys.argv[3] == 'train':
        nrmsModel.train()
        writer = SummaryWriter('data/tb/{}/{}/{}/'.format(hparams['name'], hparams['mode'], datetime.now().strftime("%Y%m%d-%H")))

    if nrmsModel.training:
        print("training...")
        loss_func = getLoss(nrmsModel)
        optimizer = optim.Adam(nrmsModel.parameters(),lr=0.001)
        nrmsModel = run_train(nrmsModel,loader_train,optimizer,loss_func,writer=writer,epochs=hparams['epochs'], interval=10)
        torch.save(nrmsModel.state_dict(), save_path)
        print("save success!")

    print("evaluating...")
    nrmsModel.eval()
    nrmsModel.cdd_size = 1
    run_eval(nrmsModel,loader_test)