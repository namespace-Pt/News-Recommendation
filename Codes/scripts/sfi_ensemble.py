import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,prepare,load_hparams,test,pipeline_encode

if __name__ == "__main__":

    hparams = {
        'name':'sfi-nrms-ensemble',
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
        hparams['encoder'] = hparams['encoder'] + '-[{}]'.format(hparams['pipeline'])
        from models.Encoders.General import Pipeline_Encoder
        encoder = Pipeline_Encoder(hparams)

    elif hparams['encoder'] == 'bert':
        hparams['encoder'] = hparams['encoder'] + '-[{}]'.format(hparams['bert'])
        from models.Encoders.General import Bert_Encoder
        encoder = Bert_Encoder(hparams)

    else:
        raise ValueError("Undefined Encoder:{}".format(hparams['encoder']))

    if hparams['interactor'] == 'fim':
        from models.Interactors import FIM_Interactor
        interactor = FIM_Interactor()

    elif hparams['interactor'] == 'knrm':
        from models.Interactors import KNRM_Interactor
        interactor = KNRM_Interactor()

    else:
        raise ValueError("Undefined Interactor:{}".format(hparams['interactor']))

    # FIXME, elasticity
    from models.SFI import SFI_ensemble, SFI_gating_MultiView
    from models.NRMS import NRMS_MultiView

    sfi = SFI_gating_MultiView(hparams, encoder, interactor).to(hparams['device']).to(hparams['device'])
    nrms = NRMS_MultiView(hparams, vocab).to(hparams['device'])

    sfi.load_state_dict(torch.load('data/model_params/sfi-multiview-fim-fim-gating/large_epoch3_step70490_[hs=50,topk=30,attrs=title,vert,subvert,abs].model',map_location=hparams['device'])['model'])
    nrms.load_state_dict(torch.load('data/model_params/nrms/large_epoch6_step33834_[hs=50,topk=0,attrs=title,vert,subvert,abs].model',map_location=hparams['device'])['model'])

    sfiModel = SFI_ensemble(hparams, (sfi,nrms))
    if hparams['mode'] == 'dev':
        evaluate(sfiModel, hparams, loaders[0])

    elif hparams['mode'] == 'test':
        test(sfiModel, hparams, loaders[0])