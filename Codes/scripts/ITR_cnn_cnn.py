import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test
from models.ITR_CNN_CNN import GCAModel

if __name__ == "__main__":

    hparams = {
        'scale':'demo',
        'name':'itr-cnn-cnn',
        'dropout_p':0.2,
        'embedding_dim':300,
        'filter_num':400,
    }
    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loaders = prepare(hparams)
    gcaModel = GCAModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'dev':
        evaluate(gcaModel,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(gcaModel, hparams, loaders)
    
    elif hparams['mode'] == 'test':
        test(gcaModel, hparams, loaders[0])