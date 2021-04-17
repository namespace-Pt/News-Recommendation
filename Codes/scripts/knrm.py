import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test,tune
from models.KNRM import KNRMModel

if __name__ == "__main__":
    hparams = {
        'name':'knrm',
        'embedding_dim':300,
        'kernel_num':11,
    }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loaders = prepare(hparams)
    knrmModel = KNRMModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'dev':
        evaluate(knrmModel,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(knrmModel, hparams, loaders)

    elif hparams['mode'] == 'test':
        test(knrmModel, hparams, loaders[0])