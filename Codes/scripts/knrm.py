import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams
from models.KNRM import KNRMModel

if __name__ == "__main__":    
    hparams = {
        'name':'knrm',
        'embedding_dim':300,
        'kernel_num':11,
        'attrs': ['title'],
    }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    knrmModel = KNRMModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'test':
        knrmModel.load_state_dict(torch.load(hparams['save_path']))
        print("testing...")
        evaluate(knrmModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(knrmModel, hparams, loader_train, loader_test, loader_validate, tb=True)