import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test

if __name__ == "__main__":

    hparams = {
        'name':'sfi',
        'dropout_p':0.2,
        'embedding_dim':300,
        'filter_num':150,
    }
    hparams = load_hparams(hparams)
    torch.cuda.set_device(hparams['device'])

    vocab, loaders = prepare(hparams)

    if hparams['encoder'] == 'fim':
        hparams['name'] = hparams['name'] + '-fim'
        from models.Encoders import FIM_Encoder
        encoder = FIM_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'mha':
        hparams['name'] = hparams['name'] + '-mha'
        hparams['value_dim'] = 16
        hparams['query_dim'] = 200
        hparams['head_num'] = 16
        from models.Encoders import MHA_Encoder
        encoder = MHA_Encoder(hparams, vocab)

    elif hparams['encoder'] == 'npa':
        hparams['name'] = hparams['name'] + '-npa'
        hparams['user_dim'] = 200
        hparams['query_dim'] = 200
        hparams['filter_num'] = 400
        from models.Encoders import NPA_Encoder
        encoder = NPA_Encoder(hparams, vocab, len(loaders[0].dataset.uid2index))

    if hparams['select'] == 'unified':
        hparams['name'] = hparams['name'] + '-unified'
        from models.SFI import SFI_unified
        sfiModel = SFI_unified(hparams, encoder).to(hparams['device'])
    
    elif hparams['select'] == 'pipeline1':
        hparams['name'] = hparams['name'] + '-pipeline1'
        from models.SFI import SFI_pipeline1
        sfiModel = SFI_pipeline1(hparams, encoder).to(hparams['device'])

    elif hparams['select'] == 'pipeline2':
        hparams['name'] = hparams['name'] + '-pipeline2'
        from models.SFI import SFI_pipeline2
        sfiModel = SFI_pipeline2(hparams, encoder).to(hparams['device'])
    
    elif hparams['select'] == 'gating':
        hparams['name'] = hparams['name'] + '-gating'
        from models.SFI import SFI_gating
        sfiModel = SFI_gating(hparams, encoder).to(hparams['device'])
    
    if hparams['mode'] == 'dev':
        evaluate(sfiModel,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(sfiModel, hparams, loaders)
    
    elif hparams['mode'] == 'test':
        test(sfiModel, hparams, loaders[0])