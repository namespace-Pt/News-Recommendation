import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams
from models.NRMS import NRMSModel

if __name__ == "__main__":

    hparams = {
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
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:0',
        'attrs': ['title'],
      }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    nrmsModel = NRMSModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'test':
        nrmsModel.load_state_dict(torch.load(hparams['save_path']))
        print("testing...")
        evaluate(nrmsModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(nrmsModel, hparams, loader_train, loader_test, loader_validate, tb=True)