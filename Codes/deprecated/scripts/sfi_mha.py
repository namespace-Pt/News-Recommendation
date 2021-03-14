import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test

if __name__ == "__main__":

    hparams = {
        'name':'sfi-mha',
        'dropout_p':0.2,
        'embedding_dim':300,

    }
    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])
    vocab, loaders = prepare(hparams)
    
    if hparams['select'] == 'pipeline':
        from models.SFI_MHA import SFIModel_pipeline
        sfiModel = SFIModel_pipeline(vocab=vocab,hparams=hparams).to(device)

    elif hparams['select'] == 'gating':
        from models.SFI_MHA import SFIModel_gating
        sfiModel = SFIModel_gating(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'dev':
        evaluate(sfiModel,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(sfiModel, hparams, loaders)
    
    elif hparams['mode'] == 'test':
        test(sfiModel, hparams, loaders[0])