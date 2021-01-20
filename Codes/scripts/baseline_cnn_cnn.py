import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams
from models.baseline_CNN_CNN import GCAModel

if __name__ == "__main__":

    hparams = {
        'scale':'demo',
        'name':'baseline-cnn-cnn',
        'batch_size':100,
        'title_size':30,
        'his_size':50,
        'npratio':4,
        'dropout_p':0.2,
        'query_dim':200,
        'embedding_dim':300,
        'filter_num':400,
        'value_dim':16,
        'head_num':16,
        'epochs':5,
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'attrs': ['title'],
    }
    hparams = load_hparams(hparams)
    
    device = torch.device(hparams['device'])

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    gcaModel = GCAModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'test':
        print("testing...")
        evaluate(gcaModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(gcaModel, hparams, loader_train, loader_test, loader_validate, tb=True)