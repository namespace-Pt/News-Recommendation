'''
Author: Pt
Date: 2020-11-16 23:54:45
LastEditTime: 2020-11-21 01:25:01
Description: 
'''
import os
import sys
os.chdir('../')
sys.path.append('../')

import torch
from utils.utils import run_eval,train,prepare
from models.FIM import FIMModel

if __name__ == "__main__":
    hparams = {
        'mode':sys.argv[1],
        'name':'fim',
        'batch_size':100,
        'title_size':18,
        'his_size':50,
        'kernel_size':3,
        'npratio':4,
        'dropout_p':0.2,
        'dilation_level':3,
        'filter_num':150,
        'embedding_dim':300,
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:0',
        'attrs': ['title','category','subcategory'],
        'epochs':int(sys.argv[3])
    }

    save_path = 'models/model_params/{}_{}_{}'.format(hparams['name'],hparams['mode'],hparams['epochs']) +'.model'
    device = torch.device(hparams['device']) if torch.cuda.is_available() else torch.device("cpu")

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    
    fimModel = FIMModel(vocab=vocab,hparams=hparams).to(device)

    if sys.argv[2] == 'eval':
        fimModel.load_state_dict(torch.load(save_path))
        fimModel.eval()
        print("evaluating...")
        run_eval(fimModel,loader_test)

    elif sys.argv[2] == 'train':
        train(fimModel, hparams, loader_train, loader_test, save_path, loader_validate, tb=True)