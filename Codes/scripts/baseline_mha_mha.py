import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams
from models.baseline_MHA_MHA import GCAModel

if __name__ == "__main__":

    hparams = {
        'name':'baseline-mha-mha',
        'batch_size':32,
        'title_size':30,
        'his_size':30,
        'npratio':4,
        'dropout_p':0.2,
        'query_dim':200,
        'embedding_dim':300,
        'value_dim':16,
        'head_num':16,
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'attrs': ['title'],
    }
    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    gcaModel = GCAModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'test':
        gcaModel.load_state_dict(torch.load(hparams['save_path']))
        print("testing...")
        evaluate(gcaModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(gcaModel, hparams, loader_train, loader_test, loader_validate, tb=True)