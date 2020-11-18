from utils.preprocess import MINDIterator
import os,torch

hparams = {
    'mode':'large',
    'batch_size':5,
    'title_size':18,
    'his_size':50,
    'kernel_size':3,
    'npratio':4,     
    'dilation_level':3,
    'filter_num':150,
    'embedding_dim':300,
    'metrics':'group_auc,ndcg@4,mean_mrr',
    'gpu':'cuda:0',
    'attrs': ['title','category','subcategory']
}
news_file_train = 'D:/Data/NR_data/dev/news_train.tsv'
news_file_test = 'D:/Data/NR_data/dev/news_test.tsv'
behavior_file_train = 'D:/Data/NR_data/dev/behaviors_train.tsv'
behavior_file_test = 'D:/Data/NR_data/dev/behaviors_test.tsv'
save_path = 'models/model_param/FIM_'+ hparams['mode'] +'.model'

# if user2id,word2id,news2id haven't been constructed
if not os.path.exists('data/nid2idx_{}_{}.json'.format(hparams['mode'],'train')):
    constructBasicDict(news_file_train,behavior_file_train,hparams['mode'],'train',hparams['attrs'])

if not os.path.exists('data/nid2idx_{}_{}.json'.format(hparams['mode'],'test')):
    constructBasicDict(news_file_test,behavior_file_test,hparams['mode'],'test',hparams['attrs'])

device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device("cpu")

iterator_train = MINDIterator(hparams=hparams,mode='train',news_file=news_file_train,behaviors_file=behavior_file_train)

iterator_test = MINDIterator(hparams=hparams,mode='test',news_file=news_file_test,behaviors_file=behavior_file_test)
iterator_test.npratio = -1
print(next(iterator_test.load_data_from_file()))