import os
import sys
os.chdir('../')
sys.path.append('../')

import torch
from utils.utils import run_eval,train,prepare
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
        'epochs':int(sys.argv[3]),
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:0',
        'attrs': ['title'],
    }

    save_path = 'models/model_params/{}_{}_{}'.format(hparams['name'],hparams['mode'],hparams['epochs']) +'.model'
    device = torch.device(hparams['device']) if torch.cuda.is_available() else torch.device("cpu")

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    nrmsModel = NRMSModel(vocab=vocab,hparams=hparams).to(device)

    if sys.argv[2] == 'eval':
        nrmsModel.load_state_dict(torch.load(save_path))
        nrmsModel.eval()
        print("evaluating...")
        run_eval(nrmsModel,loader_test)

    elif sys.argv[2] == 'train':
        train(nrmsModel, hparams, loader_train, loader_test, save_path, loader_validate, tb=True)