import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import re
from utils.utils import evaluate,train,prepare,load_hparams,test,tune,load,encode
from models.FIM import FIMModel

if __name__ == "__main__":
    hparams = {
        'name':'fim',
        'dropout_p':0.2,
        'filter_num':150,
        'embedding_dim':300,
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

    fimModel = FIMModel(hparams, encoder).to(hparams['device'])

    if re.search('pipeline', fimModel.encoder.name):
        hparams['name'] = hparams['pipeline']
    else:
        hparams['name'] = '-'.join([hparams['name'], hparams['encoder']])

    if hparams['mode'] == 'dev':
        evaluate(fimModel,hparams,loaders[0],loading=True)

    elif hparams['mode'] == 'train':
        train(fimModel, hparams, loaders)

    elif hparams['mode'] == 'test':
        test(fimModel, hparams, loaders[0])

    elif hparams['mode'] == 'encode':
        from models.Encoders.General import Encoder_Wrapper
        encoder_wrapper = Encoder_Wrapper(hparams, encoder).to('cpu').eval()

        load(encoder_wrapper, hparams, hparams['epochs'], hparams['save_step'][0])
        encode(encoder_wrapper, hparams, loader=loaders[1])
        # pipeline_encode(encoder_wrapper, hparams, loaders)