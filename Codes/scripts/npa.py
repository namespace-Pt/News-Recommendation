import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams
from models.NPA import NPAModel

if __name__ == "__main__":
    hparams = {
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
        'attrs': ['title'],
    }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    npaModel = NPAModel(vocab=vocab,hparams=hparams,uid2idx=loader_train.dataset.uid2idx).to(device)
    
    if hparams['mode'] == 'test':
        print("testing...")
        evaluate(npaModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(npaModel, hparams, loader_train, loader_test, loader_validate, tb=True)