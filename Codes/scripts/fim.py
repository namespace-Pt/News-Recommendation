import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test,getVocab
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

    if hparams['mode'] != 'submit':
        vocab, loader_train, loader_test = prepare(hparams, validate=False)

    else:
        from torchtext.vocab import GloVe
        vocab = getVocab('data/dictionaries/vocab_{}_{}.pkl'.format(hparams['scale'],'_'.join(hparams['attrs'])))
        embedding = GloVe(dim=300,cache='.vector_cache')
        vocab.load_vectors(embedding)

    fimModel = FIMModel(vocab=vocab,hparams=hparams).to(device)
    
    if hparams['mode'] == 'test':
        fimModel.load_state_dict(torch.load(hparams['save_path']))
        print("testing...")
        evaluate(fimModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        if hparams['validate']:
            train(fimModel, hparams, loader_train, loader_test, tb=True)
        else:
            train(fimModel, hparams, loader_train, loader_test, tb=True)

    elif hparams['mode'] == 'submit':
        test(fimModel, hparams)