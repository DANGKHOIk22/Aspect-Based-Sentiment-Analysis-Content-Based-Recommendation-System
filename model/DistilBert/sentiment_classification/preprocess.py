from datasets import load_dataset
import torch
from transformers import AutoTokenizer
def tokenize(corpus):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    return tokenizer

def tokenize_and_align_labels(examples):
    sentences, sentence_tags = [], []
    labels = []
    for tokens, pols in zip(examples['Tokens'],examples['Polarities']):
        bert_tokens = []
        bert_att = []
        pols_label = 0
        
        for i in range(len(tokens)):
            t = tokenizer(tokens[i])
            bert_tokens += t
            
            if int(pols[i]) != -1:
                pols_label = int(pols[i])
                bert_att += t
                
        sentences.append(' '.join(bert_tokens))
        sentence_tags.append(' '.join(bert_att))
        labels.append(pols_label)
            
    tokenized_inputs = tokenizer(sentences, sentence_tags, padding=True, truncation=True, return_tensors="pt")
    tokenized_inputs['labels'] = labels
    return tokenized_inputs
                
def preprocess_pipeline():
    ds = load_dataset("thainq107/abte-restaurants")

    max_len_tokens = max([len(tokens) for tokens in ds['train']['Tokens'] ])
    max_len_tags = max([ len([token  for token in tokens if token != '0']) for tokens in ds['train']['Tags'] ])
    global MAX_LEN
    MAX_LEN = max_len_tokens + max_len_tags

    global tokenizer
    tokenizer = tokenize()

    preprocessed_ds = ds.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing and aligning labels"
    )
    
    return preprocessed_ds