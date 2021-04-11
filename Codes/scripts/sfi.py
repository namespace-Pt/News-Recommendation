import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test,load,pipeline_encode

if __name__ == "__main__":

    hparams = {
        'name':'sfi',
        'dropout_p':0.2,
        'embedding_dim':300,
        'filter_num':150,
    }
    hparams = load_hparams(hparams)
    torch.cuda.set_device(hparams['device'])

    if hparams['mode'] == 'encode':
        vocab, loaders = prepare(hparams, news=True)
    else:
        vocab, loaders = prepare(hparams)

    if hparams['encoder'] == 'fim':
        from models.Encoders import FIM_Encoder
        encoder = FIM_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'mha':
        hparams['value_dim'] = 16
        hparams['query_dim'] = 200
        hparams['head_num'] = 16
        from models.Encoders import MHA_Encoder
        encoder = MHA_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'npa':
        hparams['user_dim'] = 200
        hparams['query_dim'] = 200
        hparams['filter_num'] = 400
        from models.Encoders import NPA_Encoder
        encoder = NPA_Encoder(hparams, vocab, len(loaders[0].dataset.uid2index))

    elif hparams['encoder'] == 'nrms':
        hparams['value_dim'] = 16
        hparams['query_dim'] = 200
        hparams['head_num'] = 16
        from models.Encoders import NRMS_Encoder
        encoder = NRMS_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'cnn':
        from models.Encoders import CNN_Encoder
        encoder = CNN_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'pipeline':
        hparams['encoder'] = hparams['encoder'] + '-[{}]'.format(hparams['pipeline'])
        from models.Encoders import Pipeline_Encoder
        encoder = Pipeline_Encoder(hparams)

    elif hparams['encoder'] == 'bert':
        hparams['encoder'] = hparams['encoder'] + '-[{}]'.format(hparams['bert'])
        from models.Encoders import Bert_Encoder
        encoder = Bert_Encoder(hparams)

    else:
        raise ValueError("Undefined Encoder:{}".format(hparams['encoder']))

    if hparams['interactor'] == 'fim':
        from models.Interactors import FIM_Interactor
        interactor = FIM_Interactor(hparams['k'], encoder.signal_length)

    elif hparams['interactor'] == 'knrm':
        from models.Interactors import KNRM_Interactor
        interactor = KNRM_Interactor()

    else:
        raise ValueError("Undefined Interactor:{}".format(hparams['interactor']))

    if hparams['select'] == 'unified':
        hparams['name'] = '-'.join([hparams['name'], hparams['encoder'], hparams['interactor'], hparams['select']])
        from models.SFI import SFI_unified
        sfiModel = SFI_unified(hparams, encoder, interactor).to(hparams['device'])

    elif hparams['select'] == 'gating':
        hparams['name'] = '-'.join([hparams['name'], hparams['encoder'], hparams['interactor'], hparams['select']])
        from models.SFI import SFI_gating
        sfiModel = SFI_gating(hparams, encoder, interactor).to(hparams['device'])

    else:
        raise ValueError("Undefined Selection Method:{}".format(hparams['select']))

    if hparams['mode'] == 'dev':
        evaluate(sfiModel,hparams,loaders[0],loading=True)

    elif hparams['mode'] == 'train' or hparams['mode'] == 'whole':
        train(sfiModel, hparams, loaders)

    elif hparams['mode'] == 'test':
        test(sfiModel, hparams, loaders[0])

    elif hparams['mode'] == 'encode':
        from models.Encoders import Encoder_Wrapper
        encoder_wrapper = Encoder_Wrapper(hparams, encoder).to(hparams['device'])
        hparams['name'] = '-'.join([hparams['name'], hparams['encoder'], hparams['select']])

        load(Encoder_Wrapper, hparams, hparams['epochs'], hparams['save_step'], pipeline=True)
        pipeline_encode(Encoder_Wrapper, hparams, loaders)
