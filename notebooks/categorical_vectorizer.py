from typing import List

import numpy as np
import pandas as pd
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD


def countvec_feature(transaction: pd.DataFrame, key: str, target: str) -> pd.DataFrame:
    """Return count vector features on 'target', groupby 'key'.
    Args:
        transaction (pd.DataFrame):Transaction data which has a key column for join to master data and target column.
        key (str): A key for groupby.
        target (str): A column name for get SVD features.
        max_features(int): Desired dimention of output data.
        
    Returns:
        Count vecotr features and key (pd.DataFrame).
    """
    count_vec = CountVectorizer()
    df_bag = pd.DataFrame(df[[key, target]])
    df_bag = df_bag.groupby(key, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))

    
    count_vector = count_vec.fit_transform(df_bag[target + '_list'])
    count_features = pd.SparseDataFrame(count_vector).fillna(0)
    count_features.columns = ['count_%s_%d'%(target,x) for x in range(10000)]
    count_features[key] = df_bag[key]
    
    return count_features


def tfidf_feature(transaction: pd.DataFrame, key: str, target: str, max_features: int=10000) -> pd.DataFrame:
    """Return tfidf features on 'target', groupby 'key'.
    Args:
        transaction (pd.DataFrame):Transaction data which has a key column for join to master data and target column.
        key (str): A key for groupby.
        target (str): A column name for get SVD features.
        max_features(int): Desired dimention of output data.
        
    Returns:
        Tfidf features and key (pd.DataFrame).
    """
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=max_features)
    df_bag = pd.DataFrame(df[[key, target]])
    df_bag = df_bag.groupby(key, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))

    
    tfidf_vector = tfidf_vec.fit_transform(df_bag[target + '_list'])
    tfidf_features = pd.SparseDataFrame(tfidf_vector).fillna(0)
    tfidf_features.columns = ['tfidf_%s_%d'%(target,x) for x in range(10000)]
    tfidf_features[key] = df_bag[key]
    
    return tfidf_features



def svd_feature(transaction: pd.DataFrame, key: str, target: str, n_comp: int)-> pd.DataFrame:
    """Return dimentionality reduction features using SVD on 'target' groupby 'key'.
    
    https://github.com/senkin13/kaggle/blob/master/elo/lgb.pyのsvd_feature()でmasterとマージしてるけどいらんくね？ってなって外してる
    
    Args:
        transaction (pd.DataFrame):Transaction data which has a key column for join to master data and target column.
        key (str): A key for groupby.
        target (str): A column name for get SVD features.
        n_comp (int): Desired dimensionality of output data.
        
    Returns:
        SVD features and key (pd.DataFrame).
    """
    tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=None) 
    df_bag = pd.DataFrame(transaction[[key, target]]) # トランザクションデータの必要な特徴だけ取り出す
    df_bag = df_bag.groupby(key, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index() # `key`ごとにトランザクションデータの中で触れた`target`をリストで集約してくる    
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))
    
    tfidf_vector = tfidf.fit_transform(df_bag[target + '_list'])
    svd_vec = TruncatedSVD(n_components=5, algorithm='arpack')
    svd_features = pd.DataFrame(svd_vec.fit_transform(tfidf_vector))

    svd_features.columns = ['svd_%s_%d'%(target,x) for x in range(n_comp)]
    svd_features[key] = df_bag[key]
    return svd_features


def lda_feature(transaction: pd.DataFrame, key: str, target: str, num_topics: int) -> pd.DataFrame:
    """Return dimentionality reduction features using LDA on 'target' groupby 'key'.
    Args:
        transaction (pd.DataFrame):Transaction data which has a key column for join to master data and target column.
        key (str): A key for groupby.
        target (str): A column name for get SVD features.
        num_topics (int): Desired dimensionality of output data.
        
    Returns:
        LDA features and key (pd.DataFrame).
    """
    df_bag = pd.DataFrame(transaction[[key, target]])
    df_bag[target] = df_bag[target].astype(str)
    df_bag[target].fillna('NAN', inplace=True)    
    df_bag = df_bag.groupby(key, as_index=False)[target].agg({'list':(lambda x: list(x))})
    df_bag['sentence'] = df_bag['list'].apply(lambda x: list(map(str,x)))
    
    docs = df_bag['sentence'].tolist() 
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(tokens) for tokens in docs]
    
    
    model = models.LdaModel(corpus,
                            num_topics=num_topics,
                            id2word=dictionary,
                            random_state=3655)
    
    topics_values = np.array(model.get_document_topics(corpus, minimum_probability=0))[:, :, 1]
    
    lda_features = pd.DataFrame(topics_values,dtype=np.float16).fillna(0.001)
    lda_features.columns = ['lda_%s_%d'%(target,x) for x in range(num_topics)]
    lda_features[key] = df_bag[key]
    return lda_features


def word2vec_feature(transaction: pd.DataFrame, key: str, target: str, size: int) -> pd.DataFrame:
    """Return dimentionality reduction features using Word2Vec on 'target' groupby 'key'.
    Args:
        transaction (pd.DataFrame):Transaction data which has a key column for join to master data and target column.
        key (str): A key for groupby.
        target (str): A column name for get Word2Vec features.
        size (int): Desired dimensionality of output data.
        
    Returns:
        Word2Vec features and key (pd.DataFrame).
    """
    df_bag = pd.DataFrame(df[[key, target]])
    df_bag[target] = df_bag[target].astype(str)
    df_bag[target].fillna('NAN', inplace=True)
    df_bag = df_bag.groupby(key, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    
    doc_list = list(df_bag['list'].values)
    model = models.Word2Vec(doc_list, size=size, window=3, min_count=1, workers=32)
    vocab_keys = list(model.wv.vocab.keys())
    w2v_array = []
    for v in vocab_keys :
        w2v_array.append(list(model.wv[v]))
    
    w2v_features = pd.DataFrame()
    w2v_features['vocab_keys'] = vocab_keys    
    w2v_features = pd.concat([w2v_features, pd.DataFrame(w2v_array)], axis=1)
    w2v_features.columns = [target] + ['w2v_%s_%d'%(target,x) for x in range(size)]
    return w2v_features
