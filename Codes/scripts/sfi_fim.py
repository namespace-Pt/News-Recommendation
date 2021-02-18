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
    device = torch.device(hparams['device'])

    vocab, loaders = prepare(hparams)

    if hparams['select'] == 'unified':
        from models.SFI_FIM import SFIModel_unified
        sfiModel = SFIModel_unified(vocab=vocab,hparams=hparams).to(device)
    
    elif hparams['select'] == 'pipeline1':
        from models.SFI_FIM import SFIModel_pipeline1
        sfiModel = SFIModel_pipeline1(vocab=vocab,hparams=hparams).to(device)

    elif hparams['select'] == 'pipeline2':
        from models.SFI_FIM import SFIModel_pipeline2
        sfiModel = SFIModel_pipeline2(vocab=vocab,hparams=hparams).to(device)
    
    elif hparams['select'] == 'gating':
        from models.SFI_FIM import SFIModel_gating
        sfiModel = SFIModel_gating(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'dev':
        state_dict = torch.load(hparams['save_path'])
        state_dict = {k:v for k,v in state_dict.items() if k not in ['news_reprs.weight','news_embeddings.weight']}
        sfiModel.load_state_dict(state_dict,strict=False)
        print("testing...")
        evaluate(sfiModel,hparams,loaders[1])

    elif hparams['mode'] == 'train':
        train(sfiModel, hparams, loaders)
    
    elif hparams['mode'] == 'test':
        print(loaders)
        test(sfiModel, hparams, loaders[0])