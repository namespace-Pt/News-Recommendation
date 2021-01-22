import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams
from models.ITR_CNN_CNN import GCAModel

if __name__ == "__main__":

    hparams = {
        'scale':'demo',
        'name':'itr-cnn-cnn',
        'dropout_p':0.2,
        'embedding_dim':300,
        'filter_num':400,
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