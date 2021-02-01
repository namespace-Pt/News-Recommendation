import os
import sys
os.chdir('/home/peitian_zhang/Codes/News-Recommendation')
sys.path.append('/home/peitian_zhang/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams

if __name__ == "__main__":

    hparams = {
        'name':'sfi-mha',
        'dropout_p':0.2,
        'embedding_dim':300,
        'value_dim':64,
        'query_dim':200,
        'head_num':8,
        'attrs': ['title']
    }
    hparams = load_hparams(hparams)
    device = torch.device(hparams['device'])
    vocab, loader_train, loader_test, loader_validate = prepare(hparams, validate=True)
    
    if hparams['select'] == 'pipeline':
        from models.SFI_MHA import SFIModel_pipeline
        sfiModel = SFIModel_pipeline(vocab=vocab,hparams=hparams).to(device)

    elif hparams['select'] == 'gating':
        from models.SFI_MHA import SFIModel_gating
        sfiModel = SFIModel_gating(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'test':
        sfiModel.load_state_dict(torch.load(hparams['save_path']))
        print("testing...")
        evaluate(sfiModel,hparams,loader_test)

    elif hparams['mode'] == 'train':
        train(sfiModel, hparams, loader_train, loader_test, loader_validate, tb=True)