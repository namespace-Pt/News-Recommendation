import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test

if __name__ == "__main__":

    hparams = {
        'name':'sfi-fim',
        'dropout_p':0.2,
        'embedding_dim':300,
        'filter_num':150,
    }
    hparams = load_hparams(hparams)
    torch.cuda.set_device(hparams['device'])

    vocab, loaders = prepare(hparams)

    if hparams['select'] == 'unified':
        hparams['name'] = hparams['name'] + '-unified'
        from models.SFI_FIM import SFIModel_unified
        sfiModel = SFIModel_unified(vocab=vocab,hparams=hparams).to(hparams['device'])
    
    elif hparams['select'] == 'pipeline1':
        hparams['name'] = hparams['name'] + '-pipeline1'
        from models.SFI_FIM import SFIModel_pipeline1
        sfiModel = SFIModel_pipeline1(vocab=vocab,hparams=hparams).to(hparams['device'])

    elif hparams['select'] == 'pipeline2':
        hparams['name'] = hparams['name'] + '-pipeline2'
        from models.SFI_FIM import SFIModel_pipeline2
        sfiModel = SFIModel_pipeline2(vocab=vocab,hparams=hparams).to(hparams['device'])
    
    elif hparams['select'] == 'gating':
        hparams['name'] = hparams['name'] + '-gating'
        from models.SFI_FIM import SFIModel_gating
        sfiModel = SFIModel_gating(vocab=vocab,hparams=hparams).to(hparams['device'])

    if hparams['mode'] == 'dev':
        evaluate(sfiModel,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(sfiModel, hparams, loaders)
    
    elif hparams['mode'] == 'test':
        test(sfiModel, hparams, loaders[0])