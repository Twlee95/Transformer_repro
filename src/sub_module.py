import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self,hidden_dimension, n_head):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.n_head = n_head
        
        self.d_k = self.hidden_dimension/self.n_head
        self.root_d_k = math.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, attn_mask=None, Decoder_casual_Mask = None): # batch,sequence_length,hidden_dimension/n_head

        dot_product = torch.matmul(q,k.transpose(-1,-2)) # [batch, head, QuerySeqLen, dk] [batch, head, KeySeqLen, dk]

        scaled_dot_product = dot_product / self.root_d_k # [batch, head, QuerySeqLen, KeySeqLen]

        #print("scaled_dot_product",scaled_dot_product.size()) # [256, 12, 12] [batch, dec seq_len, enc seq_len]
        # scaled_dot_product torch.Size([20, 74, 63]) B h Q K
        #print(attn_mask.size())  # B K 1 -> B K Q-> B (h) Q K

        if Decoder_casual_Mask is not None:
            attn_mask = attn_mask | Decoder_casual_Mask.expand(attn_mask.size(0), -1, -1).unsqueeze(1).repeat(1,self.n_head,1,1) # B (h) Q K

        if attn_mask is not None:
            scaled_dot_product.masked_fill_(attn_mask, -float('inf'))

        # Softmax 결과 확인
        attention_score = self.softmax(scaled_dot_product)
        attention_score = torch.nan_to_num(attention_score, nan=0.0)

        attention_value = torch.matmul(attention_score,v) #  B (h) Q K   [batch, head, k, dk]  # b,h,q,dk

        return attention_value


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dimension,embedding_dim, n_head,dropout_rate):
        super().__init__()
        
        self.n_head = n_head
        self.hidden_dimension = hidden_dimension
        self.d_k = int(self.hidden_dimension/self.n_head)
        self.embedding_dim = embedding_dim
        
        self.Q_ = nn.Linear(self.embedding_dim, self.d_k*self.n_head)
        self.K_ = nn.Linear(self.embedding_dim, self.d_k*self.n_head)
        self.V_ = nn.Linear(self.embedding_dim, self.d_k*self.n_head)

        self.scaled_dot_product_attenion = ScaledDotProductAttention(self.hidden_dimension, self.n_head)
        self.out_linear = nn.Linear(self.d_k*self.n_head,hidden_dimension)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        init.xavier_uniform_(self.Q_.weight)
        init.zeros_(self.Q_.bias)
        init.xavier_uniform_(self.K_.weight)
        init.zeros_(self.K_.bias)
        init.xavier_uniform_(self.V_.weight)
        init.zeros_(self.V_.bias)
        init.xavier_uniform_(self.out_linear.weight)
        init.zeros_(self.out_linear.bias)
    def forward(self, q_, k_, v_, attn_mask=None, dec_casual_mask = None):
        
        batch_size = q_.size(0)
        
        q = self.Q_(q_).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        k = self.K_(k_).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        v = self.V_(v_).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        
        attention = self.scaled_dot_product_attenion(q,k,v, attn_mask,dec_casual_mask) # ([20, 8, 45, 64]) # b,h,q,dk
        
        concat_attention = attention.transpose(1,2).contiguous().view(batch_size,-1, self.n_head*self.d_k) # [batch, 9, 64]
        # b,q,h*dk
        MHA_output= self.out_linear(concat_attention) 

        return self.dropout(MHA_output)
    


class FeedForwardNetwork(nn.Module):
    def __init__(self,hidden_dimension,h_ff,dropout_rate):
        super().__init__()
        
        # self.ff1 = nn.Conv1d(in_channels=hidden_dimension,out_channels=h_ff,kernel_size=1)
        # self.ff2 = nn.Conv1d(in_channels=h_ff,out_channels=hidden_dimension,kernel_size=1)

        self.ff1 = nn.Linear(hidden_dimension, h_ff)
        self.ff2 = nn.Linear(h_ff, hidden_dimension)

        # ff1 가중치 및 편향 초기화
        init.xavier_uniform_(self.ff1.weight)
        init.zeros_(self.ff1.bias)
        
        # ff2 가중치 및 편향 초기화
        init.xavier_uniform_(self.ff2.weight)
        init.zeros_(self.ff2.bias)

        
        self.dropout = nn.Dropout(dropout_rate)
        
        # # ff1 초기화
        # init.kaiming_uniform_(self.ff1.weight, nonlinearity='relu')
        # if self.ff1.bias is not None:
        #     init.zeros_(self.ff1.bias)

        # # ff2 초기화
        # init.kaiming_uniform_(self.ff2.weight, nonlinearity='relu')
        # if self.ff2.bias is not None:
        #     init.zeros_(self.ff2.bias)

    def forward(self, MHA): # [batch, seq_len, feature]

        # FF_MHA = self.ff1(MHA.transpose(1,2)) #[batch, feature*4, seq_len]
        # relu_FF_MHA = F.relu(FF_MHA)
        # ff2_output = self.ff2(relu_FF_MHA).transpose(1,2) # [batch, feature, seq_len]
        
        FF_MHA = self.ff1(MHA) #[batch, seq_len, feature*4, ]
        relu_FF_MHA = F.relu(FF_MHA)
        ff2_output = self.ff2(relu_FF_MHA) # [batch, seq_len, feature]

        return self.dropout(ff2_output)