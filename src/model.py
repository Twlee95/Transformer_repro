import torch
import torch.nn as nn
from torch.nn import init
import config
import torch.nn.functional as F
from module import TransformerEncoder, TransformerDecoder
import math

        

class Transformer(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dimension, n_head,n_layers,h_ff,dropout_rate, device):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device) #     criterion = nn.CrossEntropyLoss(label_smoothing = config.label_smoothing_rate, ignore_index = config.PAD)
        torch.nn.init.xavier_normal_(self.embedding.weight)
        self.transformer_encoder = TransformerEncoder(hidden_dimension, embedding_dim, n_head,n_layers,h_ff,dropout_rate,device)
        self.transformer_decoder = TransformerDecoder(hidden_dimension, embedding_dim, n_head,n_layers,h_ff,dropout_rate,device)
        self.fc_out = nn.Linear(hidden_dimension, vocab_size)
        self.fc_out.weight = self.embedding.weight
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout_rate)
    
        self.device = device

    def forward(self, src, tgt):
        # Enocder
        # Enc attention mask & Embedding
        enc_attn_mask = self.gen_att_mask(src,config.PAD)
        embed_src = self.embedding(src)* math.sqrt(self.embedding_dim)
        pos_src = self.Positional_Embedding(embed_src,self.device).unsqueeze(0) 
        enc_embed_input = self.dropout(embed_src+pos_src)
        
        # Ecoder_model
        encoder_output = self.transformer_encoder(enc_embed_input,enc_attn_mask)
        
        # Decoder
        # Dec attetion mask & Embedding
        dec_attn_mask = self.gen_att_mask(tgt,config.PAD)
        embed_tgt = self.embedding(tgt)* math.sqrt(self.embedding_dim)
        pos_tgt = self.Positional_Embedding(embed_tgt,self.device).unsqueeze(0)
        dec_embed_input = self.dropout(embed_tgt+pos_tgt)
        
        # Deoder_model
        decoder_output = self.transformer_decoder(encoder_output, dec_embed_input,enc_attn_mask,dec_attn_mask)
        
        return self.fc_out(decoder_output)
    
    def gen_att_mask(self,x, pad):
        a = (x==pad).unsqueeze(2)
        #print("########",a.size())
        return a
    
    
    def Positional_Embedding(self, embedding,device):
        #print(embedding.size()) # [batch, seq_len, embed_dim]
        seq_len = embedding.size(1)
        embed_dim = embedding.size(2)
        
        pos = torch.arange(seq_len,device = device).unsqueeze(1)
        i_ = torch.arange(embed_dim,device = device).unsqueeze(0)
        
        si_ = torch.pow(10000.0, (2*(i_//2))/embed_dim)
        
        pos_encoding = torch.zeros(seq_len,embed_dim,device = device)
        pos_encoding[:,0::2] = torch.sin(pos/(si_[:,0::2]))
        pos_encoding[:,1::2] = torch.cos(pos/(si_[:,1::2]))
        
        return pos_encoding
