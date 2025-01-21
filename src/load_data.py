import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import config

def read_languages(data_path, data_type):
    
    if data_type == 'train':
        source_data_path = data_path + 'train.en'
        target_data_path = data_path + 'train.de'
    if data_type == 'dev':
        source_data_path = data_path + 'newstest2014.en'
        target_data_path = data_path + 'newstest2014.de'
        
    with open(source_data_path, 'r', encoding='utf-8') as f1, open(target_data_path, 'r', encoding='utf-8') as f2:
        pairs = [[en_line.strip(), de_line.strip()] for en_line, de_line in zip(f1,f2)]
    return pairs

# def filter_pairs(pair,max_len):
#     good_pair = (3<=len(pair[0].split(" "))<= max_len) and (3<=len(pair[1].split(" "))<=max_len-1)
#     return good_pair
    
def filter_pairs(pair,max_len,tokenizer):
    good_pair = (3<=len(tokenizer.encode(pair[0]).ids)<= max_len) and (3<=len(tokenizer.encode(pair[1]).ids)<=max_len-1)
    return good_pair
    
def tokenize(tokenizer,pair,is_decoder):
    tokens = tokenizer.encode(pair).ids
    if is_decoder:
        tokens = tokenizer.encode("<s>").ids + tokens + tokenizer.encode("</s>").ids
    
    return tokens

def select_tokenize_pairs(pairs,max_len,token_path):
    ## tokenizer load
    print("\nLoading saved tokenizer...")
    tokenizer = Tokenizer.from_file(token_path)
    
    filtered_pairs = [[tokenize(tokenizer, pair[0], False), tokenize(tokenizer, pair[1], True)] for pair in pairs if filter_pairs(pair,max_len,tokenizer)]    
    return filtered_pairs,tokenizer

def prepare_data(data_path=None, token_path=None, data_type='train', max_len=100):
    pairs = read_languages(data_path, data_type)
    pairs, tokenizer = select_tokenize_pairs(pairs, max_len,token_path)
    print("length of the pairs", len(pairs))
    return pairs, tokenizer



