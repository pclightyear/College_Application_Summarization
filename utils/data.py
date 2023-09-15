# Utility variable
import sys
sys.path.insert(0, '../..')

import var.path as P
import pandas as pd
import pickle
import joblib
import os
import json

"""
Applications
"""
def read_df_applications():
    with open(P.FP_DF_APPLICATIONS_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_df_applications(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_APPLICATIONS_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_APPLICATIONS_CSV, index=False)

def read_df_applicants(train_or_test='all'):
    with open(P.FP_DF_APPLICANTS_PKL, 'rb') as fp:
        df = pickle.load(fp)
        
    if train_or_test == 'all':
        return df
    elif train_or_test == 'train':
        return df[df['train_or_test'] == 'train']
    elif train_or_test == 'test':
        return df[df['train_or_test'] == 'test']
    
def write_df_applicants(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_APPLICANTS_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_APPLICANTS_CSV, index=False)

def read_df_achievements():
    with open(P.FP_DF_ACHIEVEMENTS_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_df_achievements(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_ACHIEVEMENTS_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_ACHIEVEMENTS_CSV, index=False)

def read_achievements():
    with open(P.FP_ACHIEVEMENTS_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_achievements(_list):
    with open(P.FP_ACHIEVEMENTS_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_achievements_embed():
    with open(P.FP_ACHIEVEMENTS_SBERT_EMBED_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_achievements_embed(_list):
    with open(P.FP_ACHIEVEMENTS_SBERT_EMBED_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_application_ocr_txt(year, _id):
    fp = os.path.join(P.D_APPLICATION, year, P.TXT_OCR_DIR, "{}.json".format(_id))
    
    with open(fp, 'r') as f:
        app_texts = json.load(f)
    
    return app_texts

def read_df_recommendation_letters():
    with open(P.FP_DF_RECOMMENDATION_LETTERS_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_df_recommendation_letters(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_RECOMMENDATION_LETTERS_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_RECOMMENDATION_LETTERS_CSV, index=False)

"""
Comments
"""
def read_split_comments_train_test_indices():
    with open(P.FP_SPLIT_COMMENTS_TRAIN_TEST_INDICES, 'rb') as fp:
        return pickle.load(fp)
    
def write_split_comments_train_test_indices(_dict):
    with open(P.FP_SPLIT_COMMENTS_TRAIN_TEST_INDICES, 'wb') as fp:
        pickle.dump(_dict, fp)

split_comments_train_test_indices = read_split_comments_train_test_indices()
        
def list_data_train_test_index_split(list_data, train_test_idx):
    assert len(list_data) == len(train_test_idx)
    
    train_set = [list_data[i] for i, train_or_test in enumerate(train_test_idx) if train_or_test == 'train']
    test_set = [list_data[i] for i, train_or_test in enumerate(train_test_idx) if train_or_test == 'test']
    
    return train_set, test_set

def df_data_train_test_index_split(df_data, train_idx, test_idx):
    assert len(df_data) == (len(train_idx) + len(test_idx))
    
    train_set = df_data.filter(items=train_idx, axis=0)
    test_set = df_data.filter(items=test_idx, axis=0)
    
    return train_set, test_set

def dataset_return_helper(_all, _train, _test, train_or_test):
    if train_or_test == 'all':
        return _all
    elif train_or_test == 'train':
        return _train
    elif train_or_test == 'test':
        return _test

def read_df_comments():
    with open(P.FP_DF_COMMENTS_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_df_comments(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_COMMENTS_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_COMMENTS_CSV, index=False)

def read_df_split_comments(train_or_test='all'):
    with open(P.FP_DF_SPLIT_COMMENTS_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_idx = split_comments_train_test_indices['split_comments']['train_idx']
    test_idx = split_comments_train_test_indices['split_comments']['test_idx']
    _train, _test = df_data_train_test_index_split(_all, train_idx, test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
        
def write_df_split_comments(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_SPLIT_COMMENTS_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_SPLIT_COMMENTS_CSV, index=False)
    
def read_split_comments(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments(_list):
    with open(P.FP_SPLIT_COMMENTS_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_split_comments_embed(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_SBERT_EMBED_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments_embed(_list):
    with open(P.FP_SPLIT_COMMENTS_SBERT_EMBED_PKL, 'wb') as fp:
        pickle.dump(_list, fp)

def read_df_split_comments_no_duplicate(train_or_test='all'):
    with open(P.FP_DF_SPLIT_COMMENTS_NO_DUPLICATE_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_idx = split_comments_train_test_indices['split_comments_no_duplicate']['train_idx']
    test_idx = split_comments_train_test_indices['split_comments_no_duplicate']['test_idx']
    _train, _test = df_data_train_test_index_split(_all, train_idx, test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
        
def write_df_split_comments_no_duplicate(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_SPLIT_COMMENTS_NO_DUPLICATE_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_SPLIT_COMMENTS_NO_DUPLICATE_CSV, index=False)
        
def read_split_comments_no_duplicate(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_NO_DUPLICATE_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments_no_duplicate']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments_no_duplicate(_list):
    with open(P.FP_SPLIT_COMMENTS_NO_DUPLICATE_PKL, 'wb') as fp:
        pickle.dump(_list, fp)

def read_split_comments_no_duplicate_articut():
    with open(P.FP_SPLIT_COMMENTS_NO_DUPLICATE_ARTICUT_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_split_comments_no_duplicate_articut(_list):
    with open(P.FP_SPLIT_COMMENTS_NO_DUPLICATE_ARTICUT_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_split_comments_no_duplicate_embed(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_NO_DUPLICATE_SBERT_EMBED_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments_no_duplicate']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments_no_duplicate_embed(_list):
    with open(P.FP_SPLIT_COMMENTS_NO_DUPLICATE_SBERT_EMBED_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_split_comments_nsp(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_NSP_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments_nsp']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments_nsp(_list):
    with open(P.FP_SPLIT_COMMENTS_NSP_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_split_comments_nsp_embed(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_NSP_SBERT_EMBED_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments_nsp']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments_nsp_embed(_list):
    with open(P.FP_SPLIT_COMMENTS_NSP_SBERT_EMBED_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_split_comments_nsp_no_duplicate(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_NSP_NO_DUPLICATE_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments_nsp_no_duplicate']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments_nsp_no_duplicate(_list):
    with open(P.FP_SPLIT_COMMENTS_NSP_NO_DUPLICATE_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_split_comments_nsp_no_duplicate_embed(train_or_test='all'):
    with open(P.FP_SPLIT_COMMENTS_NSP_NO_DUPLICATE_SBERT_EMBED_PKL, 'rb') as fp:
        _all = pickle.load(fp)
    
    train_test_idx = split_comments_train_test_indices['split_comments_nsp_no_duplicate']['train_test_list']
    _train, _test = list_data_train_test_index_split(_all, train_test_idx)
    
    return dataset_return_helper(_all, _train, _test, train_or_test)
    
def write_split_comments_nsp_no_duplicate_embed(_list):
    with open(P.FP_SPLIT_COMMENTS_NSP_NO_DUPLICATE_SBERT_EMBED_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_df_tokenized_comments():
    with open(P.FP_DF_TOKENIZED_COMMENTS_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_df_tokenized_comments(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_TOKENIZED_COMMENTS_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_TOKENIZED_COMMENTS_CSV, index=False)

def read_comment_sentiment(train_or_test):
    if train_or_test == 'train':
        with open(P.FP_TRAIN_COMMENT_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    elif train_or_test == 'test':
        with open(P.FP_TEST_COMMENT_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    elif train_or_test == 'all':
        with open(P.FP_ALL_COMMENT_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    
def write_comment_sentiment(data, train_or_test):
    if train_or_test == 'train':
        with open(P.FP_TRAIN_COMMENT_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp)
    elif train_or_test == 'test':
        with open(P.FP_TEST_COMMENT_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp)
    elif train_or_test == 'all':
        with open(P.FP_ALL_COMMENT_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp)   

def read_aug_comment_sentiment(train_or_test):
    if train_or_test == 'train':
        with open(P.FP_TRAIN_AUG_COMMENT_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    elif train_or_test == 'test':
        with open(P.FP_TEST_AUG_COMMENT_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    elif train_or_test == 'all':
        with open(P.FP_ALL_AUG_COMMENT_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    
def write_aug_comment_sentiment(data, train_or_test):
    if train_or_test == 'train':
        with open(P.FP_TRAIN_AUG_COMMENT_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp)
    elif train_or_test == 'test':
        with open(P.FP_TEST_AUG_COMMENT_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp)
    elif train_or_test == 'all':
        with open(P.FP_ALL_AUG_COMMENT_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp) 
    
def read_split_comments_sentiment(train_or_test):
    if train_or_test == 'train':
        with open(P.FP_TRAIN_SPLIT_COMMENTS_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    elif train_or_test == 'all':
        with open(P.FP_ALL_SPLIT_COMMENTS_SENTIMENT_PKL, 'rb') as fp:
            return pickle.load(fp)
    
def write_split_comments_sentiment(data, train_or_test):
    if train_or_test == 'train':
        with open(P.FP_TRAIN_SPLIT_COMMENTS_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp)
    elif train_or_test == 'all':
        with open(P.FP_ALL_SPLIT_COMMENTS_SENTIMENT_PKL, 'wb') as fp:
            pickle.dump(data, fp)   
    
"""
Dictionary
"""         
def read_df_competition_dict():
    with open(P.FP_DF_COMPETITION_DICT_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_df_competition_dict(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_COMPETITION_DICT_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_COMPETITION_DICT_CSV, index=False)

def read_competition_dict_items():
    with open(P.FP_COMPETITION_DICT_ITEMS_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_competition_dict_items(_list):
    with open(P.FP_COMPETITION_DICT_ITEMS_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
def read_competition_dict_items_embed():
    with open(P.FP_COMPETITION_DICT_ITEMS_SBERT_EMBED_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_competition_dict_items_embed(_list):
    with open(P.FP_COMPETITION_DICT_ITEMS_SBERT_EMBED_PKL, 'wb') as fp:
        pickle.dump(_list, fp)
        
"""
Clustering
"""
def save_comment_clustering_model(dimension_reduction_pipe, cluster, visualization_pipe, method):
    if method == "hdbscan":
        joblib.dump((
            dimension_reduction_pipe, cluster, visualization_pipe
        ), P.FP_COMMENT_CLUSTERING_HDBSCAN_PACKAGE, compress=1)
    elif method == "bisecting_kmeans":
        joblib.dump((
            dimension_reduction_pipe, cluster, visualization_pipe
        ), P.FP_COMMENT_CLUSTERING_BISECTING_KMEANS_PACKAGE, compress=1)
        
def read_comment_clustering_model(method):
    if method == "hdbscan":
        dimension_reduction_pipe, cluster, visualization_pipe = \
            joblib.load(P.FP_COMMENT_CLUSTERING_HDBSCAN_PACKAGE)
    elif method == "bisecting_kmeans":
        dimension_reduction_pipe, cluster, visualization_pipe = \
            joblib.load(P.FP_COMMENT_CLUSTERING_BISECTING_KMEANS_PACKAGE)
    
    return dimension_reduction_pipe, cluster, visualization_pipe
        
"""
Output
"""
def read_df_summary():
    with open(P.FP_DF_SUMMARY_PKL, 'rb') as fp:
        return pickle.load(fp)
    
def write_df_summary(df, file='pkl'):
    if file == 'pkl':
        with open(P.FP_DF_SUMMARY_PKL, 'wb') as fp:
            pickle.dump(df, fp)
    elif file == 'csv':
        df.to_csv(P.FP_DF_SUMMARY_CSV, index=False)