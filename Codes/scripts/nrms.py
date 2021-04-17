import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test,tune
from models.NRMS import NRMSModel

if __name__ == "__main__":

    hparams = {
        'name':'nrms',
        'dropout_p':0.2,
        'query_dim':200,
        'embedding_dim':300,
        'value_dim':16,
        'head_num':16,
      }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loaders = prepare(hparams)
    nrmsModel = NRMSModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'dev':
        evaluate(nrmsModel,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(nrmsModel, hparams, loaders)

    elif hparams['mode'] == 'test':
        test(nrmsModel, hparams, loaders[0])

    elif hparams['mode'] == 'tune':
        tune(nrmsModel, hparams, loaders)