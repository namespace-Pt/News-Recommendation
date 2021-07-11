import os
import sys
os.chdir('./')
sys.path.append('./')

from utils.utils import prepare,load_hparams
from models.NPA import NPAModel

if __name__ == "__main__":
    hparams = {
        'name':'npa',
        'dropout_p':0.2,
        'filter_num':400,
        'embedding_dim':300,
        'user_dim':50,
        'preference_dim':200,
    }

    hparams = load_hparams(hparams)

    vocab, loaders = prepare(hparams)

    hparams['user_dim'] = 200
    hparams['query_dim'] = 200
    hparams['filter_num'] = 400
    from models.Encoders.NPA import NPA_Encoder
    encoder = NPA_Encoder(hparams, vocab, 876956)

    npaModel = NPAModel(vocab=vocab,hparams=hparams,encoder=encoder).to(hparams['device'])

    if hparams['mode'] == 'dev':
        npaModel.evaluate(hparams,loaders[0],loading=True)

    elif hparams['mode'] == 'train':
        npaModel.fit(hparams, loaders)

    elif hparams['mode'] == 'test':
        npaModel.test(hparams, loaders[0])