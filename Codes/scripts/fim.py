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
        'dropout_p':0.2,
        'filter_num':150,
        'embedding_dim':300,
        'attrs': ['title'],
    }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    fimModel = FIMModel(vocab=vocab,hparams=hparams).to(device)
    
    if hparams['mode'] == 'test':
        fimModel.load_state_dict(torch.load(hparams['save_path']))
        print("testing...")
        evaluate(fimModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(fimModel, hparams, loader_train, loader_test, loader_validate, tb=True)