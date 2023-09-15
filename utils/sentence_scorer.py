import os
import pandas as pd
import numpy as np
import random
from collections import Counter
from itertools import chain

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import torch
from torch.utils.data import DataLoader
from torch import Tensor

# Utility variable
import sys
sys.path.insert(0, '../..')

# var
import var.path as P

# utils
import utils.data as D
import utils.io as IO
import utils.torch as Tor ## BatchSentenceDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NLI_MODEL_NAME = 'joeddav/xlm-roberta-large-xnli'
SBERT_MODEL_NAME = 'ckiplab/bert-base-chinese'

GPU_NUM = 0
device = torch.device(GPU_NUM)

nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)

sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

def gpu_setting():
    nli_model.to(device)
    sbert_model.to(device)

def nli_inference(premises_hypotheses_pair, batch_size=128, remove_neutral=False, return_logits=False):
    dataset = Tor.BatchSentenceDataset(premises_hypotheses_pair)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch
    )
    
    logits_batch = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            try:
                encoding = nli_tokenizer(batch, padding=True, return_tensors='pt', truncation='only_first')
            except:
                encoding = nli_tokenizer(batch, padding=True, return_tensors='pt', truncation='longest_first')

            for key in encoding:
                if isinstance(encoding[key], Tensor):
                    encoding[key] = encoding[key].to(device)

            logits = nli_model(**encoding)[0] ## [[contradiction, neutral, entailment]]
            if remove_neutral:
                logits = logits[:, [0, 2]]
            logits_batch.append(logits)

    ## concatenate all batch result together
    logits = torch.cat(logits_batch)
    if return_logits:
        return logits
    
    probs = logits.softmax(dim=1)
    probs = probs.cpu().numpy()
    
    return probs

def claim_score(sent, batch, batch_size=128, remove_neutral=False, debug=False):
    premises_hypothesis_pair = [(sent, b) for b in batch]
    
    probs = nli_inference(premises_hypothesis_pair, batch_size, remove_neutral)
    pred_entailment = np.argmax(probs, axis=1)
    
    n = len(batch)
    cnt = Counter(pred_entailment)
    
    num_contridiction = cnt[0]
    contridiction_rate = num_contridiction / n * 1.0
    
    if remove_neutral:
        num_neutral = cnt[1]
        neutral_rate = num_neutral / n * 1.0
        num_entail = cnt[2]
        entail_rate = num_entail / n * 1.0    
    else:
        num_entail = cnt[1]
        entail_rate = num_entail / n * 1.0    
    
    if debug:
        print("Sentence: {}".format(sent))
        print("Contridiction Rate: {:.3f}".format(contridiction_rate))
        if not remove_neutral:
            print("Neutral Rate: {:.3f}".format(neutral_rate))
        print("Entail Rate: {:.3f}".format(entail_rate))

    return entail_rate

def future_plan_score(sents, batch_size=128, remove_neutral=True):
    hypothesis = "這是讀書計劃。"
    premises_hypotheses_pair = [(s, hypothesis) for s in sents]
        
    probs = nli_inference(premises_hypotheses_pair, batch_size, remove_neutral)
        
    return probs[:, -1]

# def topic_match_score(sents, topic_class_labels, batch_size=128, remove_neutral=True):
#     hypotheses = ["這是{}。".format("或".join(labels)) for labels in topic_class_labels]
#     premises_hypotheses_pair = [(s, hypothesis) for s, hypothesis in zip(sents, hypotheses)]
    
#     probs = nli_inference(premises_hypotheses_pair, batch_size, remove_neutral)
        
#     return probs[:, -1]

from sklearn.metrics.pairwise import cosine_similarity

def topic_match_score(sents, reps, k=3, batch_size=128, remove_neutral=True):
    assert len(sents) == len(reps)
    
    sents_buf = []
    for s in sents:
        for i in range(k):
            sents_buf.append(s)
            
    reps_buf = list(chain.from_iterable(reps))
    
    sents_buf_embed = sbert_model.encode(sents_buf, batch_size=128, show_progress_bar=False)
    reps_buf_embed = sbert_model.encode(reps_buf, batch_size=128, show_progress_bar=False)
    
    sim = cosine_similarity(sents_buf_embed, reps_buf_embed).diagonal().reshape(-1, 3)
    avg_sim = sim.mean(axis=1)
    
    assert len(sents) == len(avg_sim)
    
    return avg_sim
    

"""
Evidence Score
"""
def lcs(X, Y):
    """
    Dynamic Programming implementation of LCS problem
    Source: https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
    """
    
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
    # end of function lcs

def calculate_pairwise_edit_distance(
    arr1, arr2, norm, sub, labels=("arr1", "arr2")
):
    """
    Only insertion and deletion is allowed.
    Using LCS to calculate editdistance
    """
    
    data_pair_list = []
    edit_distance_list = []
    
    for s1 in arr1:
        for s2 in arr2:
            data_pair = ("".join(s1), "".join(s2))
                
            if sub:
                edit_distance = editdistance.eval(s1, s2)
            else:
                edit_distance = len(s1) + len(s2) - 2 * lcs(s1, s2)

#             norm_edit_distance = edit_distance / len(s1)
#             norm_edit_distance = edit_distance / min(len(s1), len(s2))
#             norm_edit_distance = edit_distance / max(len(s1), len(s2))
            norm_edit_distance = edit_distance / (len(s1) + len(s2))
            
            data_pair_list.append(data_pair)
            if norm:
                edit_distance_list.append(norm_edit_distance)
            else:
                edit_distance_list.append(edit_distance)
            
    df = pd.DataFrame({
        labels[0]: [p[0] for p in data_pair_list],
        labels[1]: [p[1] for p in data_pair_list],
        "edit_distance": edit_distance_list
    })
    
    df = df.sort_values(["edit_distance", labels[0], labels[1]], ascending=True)
    df = df.reset_index(drop=True)
    
    return df

def get_decay_weight_series(n):
    weight_series = []
    
    """
    series: 1 + (1/2) + (1/4) + ... + (1/2 ** n) = 2
    """
    
    for i in range(n):
        weight = 1 / (2 ** i)
        weight_series.append(weight)
        
    return weight_series

def evidence_score(sents, evidences):
    ## calculate edit distance
    df = calculate_pairwise_edit_distance(
        sents, evidences, norm=True, sub=False, labels=("sents", "evidences")
    )
    ## calculate evidence score
    df['evidence_score'] = df['edit_distance'].apply(lambda v: 1 - v)
    
    ## for each sent, find the max evidence score
    df_group = df.groupby(['sents'])
    scores = []
    
    for sent in sents:
        try:
            g = df_group.get_group(sent)
            weights = get_decay_weight_series(g.shape[0])
            score = min(1, sum(np.multiply(weights, g['evidence_score'])))
            scores.append(score)
        except:
            ## if no any evidence
            scores.append(0)

    return np.array(scores)