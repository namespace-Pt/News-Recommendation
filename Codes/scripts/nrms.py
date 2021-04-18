import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

from utils.utils import evaluate,train,prepare,load_hparams,test,tune
from models.NRMS import NRMS,NRMS_MultiView

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

    vocab, loaders = prepare(hparams)
    if 'multiview' in hparams:
        nrms = NRMS_MultiView(hparams, vocab).to(hparams['device'])
    else:
        nrms = NRMS(vocab=vocab,hparams=hparams).to(hparams['device'])

    if hparams['mode'] == 'dev':
        evaluate(nrms,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(nrms, hparams, loaders)

    elif hparams['mode'] == 'test':
        test(nrms, hparams, loaders[0])

    elif hparams['mode'] == 'tune':
        tune(nrms, hparams, loaders)