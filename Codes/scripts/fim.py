import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams
from models.FIM import FIMModel

if __name__ == "__main__":
    hparams = {
        'name':'fim',
        'batch_size':100,
        'title_size':20,
        'his_size':50,
        'kernel_size':3,
        'npratio':4,
        'dropout_p':0.2,
        'dilation_level':3,
        'filter_num':150,
        'embedding_dim':300,
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'attrs': ['title'],
    }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    fimModel = FIMModel(vocab=vocab,hparams=hparams).to(device)
    
    if hparams['mode'] == 'test':
        print("testing...")
        evaluate(fimModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(fimModel, hparams, loader_train, loader_test, loader_validate, tb=True)