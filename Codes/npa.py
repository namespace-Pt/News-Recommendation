import torch
import torch.optim as optim
from torchtext.vocab import FastText
from utils.preprocess import MINDIterator
from utils.utils import getVocab,getLoss,getLabel
from models.NPA import NPAModel
    
news_file = r'D:\Data\NR_data\MINDsmall_train\news.tsv'
behavior_file = r'D:\Data\NR_data\dev\behaviors.tsv'
hparams = {
    'mode':'small',
    'batch_size':5,     #100
    'title_size':20,    #30
    'his_size':20,      #50
    'npratio':4,        #4
    'dropout_p':0.2,
    'filter_num':400,
    'embedding_dim':300,
    'user_dim':50,
    'preference_dim':200,
    'metrics':'auc',
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
iterator = MINDIterator(hparams=hparams)

# torchtext.Vocab.vocab object
vocab = iterator.word_dict
embedding = FastText('simple',cache='.vector_cache')
vocab.load_vectors(embedding)

npaModel = NPAModel(vocab=vocab,hparams=hparams)

# migrate the model to GPU
npaModel.to(device)

loss_func = getLoss(npaModel)
optimizer = optim.Adam(npaModel.parameters(),lr=0.002)

num = 0
for epoch in range(10):
    train = iterator.load_data_from_file(news_file,behavior_file)

    for x in train:
        pred = npaModel(x)
        label = getLabel(npaModel,x)
        loss = loss_func(pred,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        num+=1
    
    print('epoch:{},loss:{}'.format(epoch,loss))