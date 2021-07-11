import os
import sys
os.chdir('./')
sys.path.append('./')

from utils.utils import prepare,load_hparams
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

    if hparams['mode'] == 'encode':
        vocab, loaders = prepare(hparams, news=True)
    else:
        vocab, loaders = prepare(hparams)

    if hparams['encoder'] == 'fim':
        from models.Encoders.FIM import FIM_Encoder
        encoder = FIM_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'mha':
        hparams['value_dim'] = 16
        hparams['query_dim'] = 200
        hparams['head_num'] = 16
        from models.Encoders.MHA import MHA_Encoder
        encoder = MHA_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'npa':
        hparams['user_dim'] = 200
        hparams['query_dim'] = 200
        hparams['filter_num'] = 400
        from models.Encoders.NPA import NPA_Encoder
        encoder = NPA_Encoder(hparams, vocab, len(loaders[0].dataset.uid2index))

    elif hparams['encoder'] == 'nrms':
        hparams['value_dim'] = 16
        hparams['query_dim'] = 200
        hparams['head_num'] = 16
        from models.Encoders.MHA import NRMS_Encoder
        encoder = NRMS_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'cnn':
        from models.Encoders.General import CNN_Encoder
        encoder = CNN_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'pipeline':
        from models.Encoders.General import Pipeline_Encoder
        encoder = Pipeline_Encoder(hparams)

    elif hparams['encoder'] == 'bert':
        hparams['encoder'] = hparams['encoder'] + '-[{}]'.format(hparams['bert'])
        from models.Encoders.General import Bert_Encoder
        encoder = Bert_Encoder(hparams)

    else:
        raise ValueError("Undefined Selection Method")

    if hparams['multiview']:
        nrms = NRMS_MultiView(hparams, vocab).to(hparams['device'])
    else:
        nrms = NRMS(vocab=vocab,hparams=hparams, encoder=encoder).to(hparams['device'])

    if hparams['mode'] == 'dev':
        nrms.evaluate(hparams,loaders[0],loading=True)

    elif hparams['mode'] == 'train':
        nrms.fit(hparams, loaders)

    elif hparams['mode'] == 'test':
        nrms.test(hparams, loaders[0])

    elif hparams['mode'] == 'tune':
        nrms.tune(hparams, loaders)

    # elif hparams['mode'] == 'encode':
    #     from models.Encoders.General import Encoder_Wrapper
    #     encoder_wrapper = Encoder_Wrapper(hparams, encoder).to('cpu').eval()

    #     load(encoder_wrapper, hparams, hparams['epochs'], hparams['save_step'][0])
    #     encode(encoder_wrapper, hparams, loader=loaders[1])
        # pipeline_encode(encoder_wrapper, hparams, loaders)