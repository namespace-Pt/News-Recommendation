import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test
from models.NPA import NPAModel

if __name__ == "__main__":
    hparams = {
        'name':'npa',
        'dropout_p':0.2,
        'filter_num':400,
        'embedding_dim':300,
        'user_dim':50,
        'preference_dim':200,
        'attrs': ['title'],
    }

    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])

    vocab, loaders = prepare(hparams)
    npaModel = NPAModel(vocab=vocab,hparams=hparams,uid2idx=loaders[0].dataset.uid2index).to(device)
    
    if hparams['mode'] == 'dev':
        npaModel.load_state_dict(torch.load(hparams['save_path']))
        print("testing...")
        evaluate(npaModel,hparams,loaders[1])

    elif hparams['mode'] == 'train':
        train(npaModel, hparams, loaders, tb=True)
    
    elif hparams['mode'] == 'test':
        test(npaModel, hparams)