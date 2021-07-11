import os
import sys
os.chdir('./')
sys.path.append('./')
from utils.utils import prepare,load_hparams

if __name__ == "__main__":

    hparams = {
        'name':'sfi',
        'dropout_p':0.2,
        'embedding_dim':300,
        'filter_num':150,
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

    elif hparams['encoder'] == 'rnn':
        from models.Encoders.General import RNN_Encoder
        encoder = RNN_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'pipeline':
        from models.Encoders.General import Pipeline_Encoder
        encoder = Pipeline_Encoder(hparams)

    elif hparams['encoder'] == 'bert':
        hparams['encoder'] = hparams['encoder'] + '-[{}]'.format(hparams['bert'])
        from models.Encoders.General import Bert_Encoder
        encoder = Bert_Encoder(hparams)

    else:
        raise ValueError("Undefined Encoder:{}".format(hparams['encoder']))

    if hparams['interactor'] == 'fim':
        from models.Interactors.FIM import FIM_Interactor
        interactor = FIM_Interactor(encoder.level, hparams['k'])

    elif hparams['interactor'] == 'knrm':
        from models.Interactors.KNRM import KNRM_Interactor
        interactor = KNRM_Interactor()

    elif hparams['interactor'] == '2dcnn':
        from models.Interactors.CNN import CNN_Interator
        interactor = CNN_Interator(hparams['k'])

    elif hparams['interactor'] == 'mha':
        from models.Interactors.MHA import MHA_Interactor
        interactor = MHA_Interactor(encoder.hidden_dim)

    else:
        raise ValueError("Undefined Interactor:{}".format(hparams['interactor']))

    if hparams['multiview']:
        if hparams['coarse']:
            from models.SFI import SFI_unified_MultiView
            sfiModel = SFI_unified_MultiView(hparams, encoder, interactor).to(hparams['device'])

        else:
            from models.SFI import SFI_MultiView
            sfiModel = SFI_MultiView(hparams, encoder, interactor).to(hparams['device'])

    else:
        if hparams['coarse']:
            from models.SFI import SFI_unified
            sfiModel = SFI_unified(hparams, encoder, interactor).to(hparams['device'])

        else:
            from models.SFI import SFI
            sfiModel = SFI(hparams, encoder, interactor).to(hparams['device'])


    if hparams['mode'] == 'dev':
        sfiModel.evaluate(hparams,loaders[0],loading=True)

    elif hparams['mode'] == 'train':
        sfiModel.fit(hparams, loaders)

    elif hparams['mode'] == 'tune':
        sfiModel.tune(hparams, loaders)

    elif hparams['mode'] == 'test':
        sfiModel.test(hparams, loaders[0])