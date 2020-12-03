import pandas
def news_token_generator_group(news_file,tokenizer,vocab,mode):
    ''' iterate news_file, collect attrs and generate them respectively
       
    Args: 
        tokenizer: torchtext.data.utils.tokenizer
        mode: int defining how many attributes to be generated
    Returns: 
        generates wordID vector of each attrs, gathered into a list
    '''
    news_df = pd.read_table(news_file,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    news_iterator = news_df.iterrows()
    
    attrs = ['title','category','subcategory','abstract']
    for _,i in news_iterator:
        result = []
        indicator = 0
        while indicator < mode:
            result.append([vocab[x] for x in tokenizer(i[attrs[indicator]])])
            indicator += 1
        yield result 