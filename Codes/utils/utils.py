import random
import re
import json
import pickle
import pandas as pd
from torchtext.data import Field
from torchtext.data import Dataset,Example

def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def newsample(news, ratio):
    """ Sample ratio samples from news list. 
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number
    
    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)

def constructVocab(news_file,save_path):
    """
        Build field using torchtext for tokenization
    
    Returns:
        torchtext.vocabulary 
    """
    news_df = pd.read_table(news_file,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    
    text_field = Field(
        tokenize='basic_english',
        lower=True
    )

    # tokenize title,abstract,category and subcategory
    preprocess_title = news_df['title'].apply(lambda x: text_field.preprocess(x))
    preprocess_abstract = news_df['abstract'].dropna().apply(lambda x: text_field.preprocess(x))
    
    # preprocess_category = news_df['category'].apply(lambda x: text_field.preprocess(x))
    # preprocess_subcategory = news_df['subcategory'].apply(lambda x: text_field.preprocess(x))

    text_field.build_vocab(preprocess_title,preprocess_abstract)
    
    output = open(save_path,'wb')
    pickle.dump(text_field.vocab,output)
    output.close()

def constructNid2idx(news_file,dic_file):
    """
        Construct news to newsID dictionary
    """
    f = open(news_file,'r',encoding='utf-8')
    nid2index = {}
    for line in f:
        nid,_,_,_,_,_,_,_ = line.strip("\n").split('\t')

        if nid in nid2index:
            continue

        nid2index[nid] = len(nid2index) + 1
    
    g = open(dic_file,'w',encoding='utf-8')
    json.dump(nid2index,g,ensure_ascii=False)
    g.close()

def constructUid2idx(behaviors_file,dic_file):
    """
        Construct user to userID dictionary
    """
    f = open(behaviors_file,'r',encoding='utf-8')
    uid2index = {}
    for line in f:
        _,uid,_,_,_ = line.strip("\n").split('\t')

        if uid in uid2index:
            continue

        uid2index[uid] = len(uid2index) + 1
    
    g = open(dic_file,'w',encoding='utf-8')
    json.dump(uid2index,g,ensure_ascii=False)
    g.close()

def constructBasicDict(news_file,behavior_file,mode):
    """construct basic dictionary

        Args:
        news_file: path of news file
        behavior_file: path of behavior file
        mode: [small/large]
    """    
    constructVocab(news_file,'./data/vocab_'+mode+'.pkl')
    constructUid2idx(behavior_file,'./data/uid2idx_'+mode+'.json')
    constructNid2idx(news_file,'./data/nid2idx_'+mode+'.json')

def getId2idx(file):
    """
        get Id2idx dictionary from json file 
    """
    g = open(file,'r',encoding='utf-8')
    dic = json.load(g)
    g.close()
    return dic

def getVocab(file):
    """
        get Vocabulary from pkl file
    """
    g = open(file,'rb')
    dic = pickle.load(g)
    g.close()
    return dic

# class DataFrameDataset(Dataset):
#    """
#        load data from pandas.DataFrame
#    """
#     def __init__(self,df:pd.DataFrame,fields:list):
#         # Datasets.__init__(examples,fields)
#         # examples: list of examples
#         # fields: List(tuple(str,Field))

#         super().__init__(
#             [
#                 # an Example is a row in training/testing
#                 Example.fromlist(list(r)[3:5],fields)

#                 # df.iterrows generates one row of DataFrame at a time
#                 # i means newsID, r means other columns
#                 # list(r) ommit keys
#                 for i,r in df.iterrows()
#             ],
#             fields
#         )
# def load_news(self,news_file):
#     # load news
#     data = pd.read_table(r'D:\Data\NR_data\MINDsmall_train\news.tsv',index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
#     text_field = getTextField(data)
#     train = DataFrameDataset(df=data.dropna(),fields=[('title',text_field),('abstract',text_field)])
#     self.news_title_array = text_field.numericalize(text_field.pad([i.title for i in train.examples]))