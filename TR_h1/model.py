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


# from torch.nn import Transformer
# class TransformerWithEmbedding(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,max_seq_len):
#         super().__init__()
#         self.src_embedding = nn.Embedding(vocab_size, d_model,padding_idx=config.PAD,device=device)  # 임베딩 레이어
#         self.tgt_embedding = nn.Embedding(vocab_size, d_model,padding_idx=config.PAD,device=device)  # 임베딩 레이어
#         #self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))  # Positional Encoding
#         self.transformer = Transformer(
#             d_model=d_model,
#             nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.fc_out = nn.Linear(d_model, vocab_size)  # Linear 레이어

#     def forward(self, src, tgt, pad_token_id):
#         # 임베딩
#         src_embedded = self.src_embedding(src)
#         tgt_embedded = self.tgt_embedding(tgt)
        
#         # Positional Encoding 추가
#         src_embedded = src_embedded + self.generate_positional_encoding(src_embedded.size(1), src_embedded.size(2), src.device)
#         tgt_embedded = tgt_embedded + self.generate_positional_encoding(tgt_embedded.size(1), tgt_embedded.size(2), tgt.device)

#         # 마스크 생성
#         src_mask = self.generate_src_mask(src, pad_token_id)  # Source Mask
#         tgt_mask = self.generate_tgt_mask(tgt.size(1))  # Target Mask
    
#         # Transformer 모델
#         transformer_output = self.transformer(
#             src_embedded, tgt_embedded, 
#             src_key_padding_mask=src_mask, tgt_mask=tgt_mask
#         )

#         # Linear 레이어로 보캡사이즈로 변환
#         logits = self.fc_out(transformer_output)
#         return logits

#     def generate_positional_encoding(self, seq_len, d_model, device):
#         """Dynamic Positional Encoding Generator"""
#         pos = torch.arange(seq_len, device=device).unsqueeze(1)
#         i = torch.arange(d_model, device=device).unsqueeze(0)

#         # Compute the scaling factor for even and odd indices
#         angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))

#         # Apply sin to even indices and cos to odd indices
#         encoding = torch.zeros(seq_len, d_model, device=device)
#         encoding[:, 0::2] = torch.sin(pos * angle_rates[:, 0::2])
#         encoding[:, 1::2] = torch.cos(pos * angle_rates[:, 1::2])

#         return encoding.unsqueeze(0)
        
#     def generate_src_mask(self, src, pad_token_id):
#         # 2-D mask 생성 (batch에 대해 동일한 src mask 사용)
#         return (src == pad_token_id)

#     def generate_tgt_mask(self, tgt_seq_len):
#         # 2-D Causal Mask 생성 (미래 정보 차단)
#         mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1)
#         return mask.masked_fill(mask == 1, float('-inf'))

#     def generate_memory_mask(self, src, tgt, pad_token_id):
#         # 3-D mask 생성 (batch별 src padding 위치를 고려)
#         return (src == pad_token_id).unsqueeze(1)