import os
import sys
os.chdir('../')
sys.path.append('../')

import torch
from utils.utils import run_eval,train,prepare
from models.KNRM import KNRMModel

if __name__ == "__main__":    
    hparams = {
        'mode':sys.argv[1],
        'name':'knrm',
        'train_embedding': False,
        'epochs':int(sys.argv[3]),
        'batch_size':100,
        'title_size':20,
        'his_size':50,
        'npratio':4,
        'embedding_dim':300,
        'kernel_num':11,
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:1',
        'attrs': ['title'],
    }

    save_path = 'models/model_params/{}_{}_{}'.format(hparams['name'],hparams['mode'],hparams['epochs']) +'.model'
    device = torch.device(hparams['device']) if torch.cuda.is_available() else torch.device("cpu")

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    
    knrmModel = KNRMModel(vocab=vocab,hparams=hparams).to(device)

    if sys.argv[2] == 'eval':
        knrmModel.load_state_dict(torch.load(save_path))
        knrmModel.eval()
        print("evaluating...")
        run_eval(knrmModel,loader_test)

    elif sys.argv[2] == 'train':
        train(knrmModel, hparams, loader_train, loader_test, save_path, loader_validate, tb=True)