import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from sub_module import MultiHeadAttention, FeedForwardNetwork


class EncoderLayer(nn.Module):
    def __init__(self,hidden_dimension, embedding_dim, n_head,h_ff,dropout_rate, device):
        super().__init__()
               
        self.multihead_attention = MultiHeadAttention(hidden_dimension, embedding_dim, n_head,dropout_rate)
        self.feedforwardnetwork = FeedForwardNetwork(hidden_dimension, h_ff,dropout_rate)
        self.n_head = n_head
        self.norm1 = nn.LayerNorm(hidden_dimension)
        self.norm2 = nn.LayerNorm(hidden_dimension)

    def forward(self,x,attn_mask=None): # batch, seq_len, hidden_dim

        attn_mask_key = attn_mask.expand(attn_mask.size(0), attn_mask.size(1), attn_mask.size(1)).unsqueeze(1).repeat(1,self.n_head,1,1).transpose(-1,-2) 
        attn_mask_query = attn_mask.expand(attn_mask.size(0), attn_mask.size(1), attn_mask.size(1)).unsqueeze(1).repeat(1,self.n_head,1,1)

        MHA_output = self.multihead_attention(x,x,x, attn_mask_key|attn_mask_query) # batch, seq_len, hidden_dim        
        FF_input = self.norm1(x+MHA_output)

        FF_output = self.feedforwardnetwork(FF_input) # batch, seq_len, hidden_dim
        layer_output = self.norm2(x+FF_output)

        return layer_output
        
class TransformerEncoder(nn.Module):
    def __init__(self,hidden_dimension,embedding_dim, n_head, n_layers,h_ff,dropout_rate, device):
        super().__init__()
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dimension, embedding_dim, n_head,h_ff,dropout_rate, device) for _ in range(n_layers)])

    def forward(self, x, attn_mask=None):
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x,attn_mask)
        
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dimension, embedding_dim, n_head,h_ff,dropout_rate,device):
        super().__init__()
        mask_ = True
        self.masked_multihead_attention = MultiHeadAttention(hidden_dimension, embedding_dim, n_head,dropout_rate)
        self.enc_dec_multihead_attention = MultiHeadAttention(hidden_dimension, embedding_dim, n_head, dropout_rate)
        self.feedforward_network = FeedForwardNetwork(hidden_dimension, h_ff,dropout_rate)
        self.n_head = n_head
        self.norm1 = nn.LayerNorm(hidden_dimension)
        self.norm2 = nn.LayerNorm(hidden_dimension)
        self.norm3 = nn.LayerNorm(hidden_dimension)
        
        self.device = device
        
    def forward(self, encoder_output, tgt, enc_attn_mask=None, dec_attn_mask=None): # tgt [batch, seq_len, 512]
        
        ## Mask
        dec_attn_mask_Query = dec_attn_mask.expand(dec_attn_mask.size(0), dec_attn_mask.size(1), dec_attn_mask.size(1)).unsqueeze(1).repeat(1,self.n_head,1,1)
        dec_attn_mask_Key = dec_attn_mask.expand(dec_attn_mask.size(0), dec_attn_mask.size(1), dec_attn_mask.size(1)).unsqueeze(1).repeat(1,self.n_head,1,1).transpose(-1,-2)
        dec_casual_mask = self.generate_casual_mask(tgt.size(1),self.device)
        
        masked_attention_out = self.masked_multihead_attention(tgt, tgt, tgt, dec_attn_mask_Query|dec_attn_mask_Key, dec_casual_mask)
        add_norm1_out = self.norm1(masked_attention_out+tgt)
        ## Mask
        enc_dec_attn_mask_key = enc_attn_mask.expand(enc_attn_mask.size(0), enc_attn_mask.size(1), dec_attn_mask.size(1)).unsqueeze(1).repeat(1,self.n_head,1,1).transpose(-1,-2)
        enc_dec_attn_mask_query = dec_attn_mask.expand(dec_attn_mask.size(0), dec_attn_mask.size(1), enc_attn_mask.size(1)).unsqueeze(1).repeat(1,self.n_head,1,1)

        enc_dec_multihead_out = self.enc_dec_multihead_attention(add_norm1_out, encoder_output, encoder_output, enc_dec_attn_mask_key|enc_dec_attn_mask_query, None)
        add_norm2_out = self.norm2(enc_dec_multihead_out+add_norm1_out)
        FF_out = self.feedforward_network(add_norm2_out)
        add_norm3_out = self.norm3(FF_out+add_norm2_out)
        return add_norm3_out
        
    def generate_casual_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dimension, embedding_dim, n_head,n_layers,h_ff,dropout_rate,device):
        super().__init__()
        
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dimension, embedding_dim, n_head,h_ff,dropout_rate,device) for _ in range(n_layers)])
    
    def forward(self, encoder_out, tgt,enc_attn_mask=None,dec_attn_mask=None):
        
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(encoder_out,tgt,enc_attn_mask,dec_attn_mask)
        return tgt

