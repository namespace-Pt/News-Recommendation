import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import run_eval,train,prepare
from models.baseline_transformer import GCAModel

if __name__ == "__main__":

    hparams = {
        'mode':sys.argv[1],
        'name':'baseline-transformer',
        'train_embedding':False,
        'save_each_epoch':True,
        'batch_size':32,
        'title_size':20,
        'his_size':30,
        'npratio':4,
        'dropout_p':0.2,
        'query_dim':200,
        'embedding_dim':300,
        'value_dim':16,
        'head_num':16,
        'epochs':int(sys.argv[3]),
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:0',
        'attrs': ['title']
    }
    save_path = 'models/model_params/{}_{}_{}'.format(hparams['name'],hparams['mode'],hparams['epochs']) +'.model'
    device = torch.device(hparams['device']) if torch.cuda.is_available() else torch.device("cpu")

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    gcaModel = GCAModel(vocab=vocab,hparams=hparams).to(device)

    if sys.argv[2] == 'eval':
        gcaModel.load_state_dict(torch.load(save_path))
        gcaModel.eval()
        print("evaluating...")
        run_eval(gcaModel,loader_test)

    elif sys.argv[2] == 'train':
        train(gcaModel, hparams, loader_train, loader_test, save_path, loader_validate, tb=True)