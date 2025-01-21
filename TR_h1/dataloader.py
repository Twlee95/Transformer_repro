import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import config
    
from itertools import islice

def collate_fn(batch):
    source_sent, target_sent, src_len, tgt_len = zip(*batch)
    
    enc_tokens = [torch.tensor(sentence) for sentence in source_sent]
    dec_tokens = [torch.tensor(sentence) for sentence in target_sent]

        
    padded_enc_tokens = pad_sequence(enc_tokens, batch_first=True, padding_value = config.PAD)
    padded_dec_tokens = pad_sequence(dec_tokens, batch_first=True, padding_value = config.PAD)
    
    return padded_enc_tokens,padded_dec_tokens, src_len, tgt_len


class TranslationDataset(Dataset):
    def __init__(self, data,max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sent = self.data[idx][0]
        tgt_sent = self.data[idx][1]
        src_len = len(src_sent)
        tgt_len = len(tgt_sent)

        # src_sent = src_sent + [config.PAD]*(self.max_len-src_len)
        # tgt_sent = tgt_sent + [config.PAD]*(self.max_len-tgt_len+1)
        
        #return torch.tensor(src_sent,dtype = torch.long), torch.tensor(tgt_sent,dtype = torch.long), src_len, tgt_len

        return src_sent, tgt_sent, src_len, tgt_len




# class DynamicBatchSampler(Sampler):
#     def __init__(self, data,max_tokens):
#         self.data = data
#         self.max_tokens = max_tokens
#         self.sorted_index =  sorted(range(len(data)), key = lambda idx: len(data[idx][0]), reverse=True)
#         self.batch_count = 0
        
#     def __iter__(self):
#         batch = []
#         current_tokens = 0
#         self.batch_count = 0
#         for idx in self.sorted_index:
#             sample_length = len(self.data[idx][0])
#             if current_tokens + sample_length > self.max_tokens:
#                 self.batch_count += 1
#                 yield batch
#                 batch = []
#                 current_tokens = 0

#             batch.append(idx)
#             current_tokens += sample_length
        
#         if batch:
#             #print(f"Yielding final batch: {batch}")  # 디버깅 출력
#             self.batch_count += 1
#             yield batch  
    
#     def __len__(self):
#         return self.batch_count
    
    
# class SequenceSampler(Sampler):
#     def __init__(self, data):
#         self.data = data
#         self.sorted_index =  sorted(range(len(data)), key = lambda idx: len(data[idx][0]), reverse=True)
        
#     def __iter__(self):
#         return iter(self.sorted_index)

#     def __len__(self):
#         return len(self.sorted_index)