# Utility variable
import sys
sys.path.insert(0, '../..')

import utils.torch as Tor ## BatchSentenceDataset
import torch
from torch import Tensor
from torch.utils.data import DataLoader

MAX_LEN = 512

def is_next_sentence(model, tokenizer, prompt, next_sentence, device='cpu', debug=False):
    encoding = tokenizer(prompt, next_sentence, truncation='longest_first', max_length=MAX_LEN, return_tensors="pt")
    
    for key in encoding:
        if isinstance(encoding[key], Tensor):
            encoding[key] = encoding[key].to(device)
    
    with torch.no_grad():
        try:
            outputs = model(**encoding)
            logits = outputs.logits
        except Exception as e:
            print("================= Inference Error ==============")
            print(e)
            print(prompt)
            print(next_sentence)
            return None
        
    prob_yes = logits[0, 0]
    prob_no = logits[0, 1]
    
    if debug:
        print("Yes: {}; No: {}".format(prob_yes, prob_no))
    
    return bool(prob_yes > prob_no)

def is_next_sentence_batch(model, tokenizer, prompt_list, next_sentence_list, device='cpu', batch_size=256, debug=False):
    assert len(prompt_list) == len(next_sentence_list)
    
    prompt_dataset = Tor.BatchSentenceDataset(prompt_list)
    prompt_dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False)
    next_sentence_dataset = Tor.BatchSentenceDataset(next_sentence_list)
    next_sentence_dataloader = DataLoader(next_sentence_dataset, batch_size=batch_size, shuffle=False)
    
    logits_batch = []
    
    with torch.no_grad():
        for prompt_batch, next_sentence_batch in zip(prompt_dataloader, next_sentence_dataloader):
            encoding = tokenizer(prompt_batch, next_sentence_batch, truncation=='longest_first', max_length=MAX_LEN, return_tensors="pt")

            for key in encoding:
                if isinstance(encoding[key], Tensor):
                    encoding[key] = encoding[key].to(device)

            outputs = model(**encoding)
            logits = outputs.logits

            logits_batch.append(logits)
    
    logits_batch = torch.cat(logits_batch)
    prob_yes_batch = logits_batch[:, 0]
    prob_no_batch = logits_batch[:, 1]
    
    if debug:
        for prob_yes, prob_no in zip(prob_yes_batch, prob_no_batch):
            print("Yes: {}; No: {}".format(prob_yes, prob_no))
    
    return (prob_yes_batch > prob_no_batch).tolist()