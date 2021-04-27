from utils.utils import evaluate,train,prepare,load_hparams,test,tune,load,pipeline_encode,encode

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

    elif hparams['encoder'] == 'pipeline':
        from models.Encoders.General import Pipeline_Encoder
        encoder = Pipeline_Encoder(hparams)

    elif hparams['encoder'] == 'bert':
        hparams['encoder'] = hparams['encoder'] + '-[{}]'.format(hparams['bert'])
        from models.Encoders.General import Bert_Encoder
        encoder = Bert_Encoder(hparams)

    else:
        raise ValueError("Undefined Encoder:{}".format(hparams['encoder']))

    if hparams['interactor'] == None:
        interactor = None
        hparams['interactor'] = 'fim'

    elif hparams['interactor'] == 'knrm':
        from models.Interactors import KNRM_Interactor
        interactor = KNRM_Interactor()

    else:
        raise ValueError("Undefined Interactor:{}".format(hparams['interactor']))

    if 'multiview' in hparams:
        hparams['name'] = 'sfi-multiview'
        if hparams['select'] == 'unified':
            hparams['name'] = '-'.join([hparams['name'], hparams['encoder'], hparams['interactor'], hparams['select']])
            from models.SFI import SFI_unified_MultiView
            sfiModel = SFI_unified_MultiView(hparams, encoder, interactor).to(hparams['device'])

        elif hparams['select'] == 'gating':
            hparams['name'] = '-'.join([hparams['name'], hparams['encoder'], hparams['interactor'], hparams['select']])
            from models.SFI import SFI_gating_MultiView
            sfiModel = SFI_gating_MultiView(hparams, encoder, interactor).to(hparams['device'])

        else:
            raise ValueError("Undefined Selection Method:{}".format(hparams['select']))
    else:
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

    elif hparams['mode'] == 'train':
        train(sfiModel, hparams, loaders)

    elif hparams['mode'] == 'tune':
        tune(sfiModel, hparams, loaders)

    elif hparams['mode'] == 'test':
        test(sfiModel, hparams, loaders[0])

    elif hparams['mode'] == 'encode':
        from models.Encoders.General import Encoder_Wrapper
        encoder_wrapper = Encoder_Wrapper(hparams, encoder).to(hparams['device'])

        load(encoder_wrapper, hparams, hparams['epochs'], hparams['save_step'][0])
        pipeline_encode(encoder_wrapper, hparams, loaders)
