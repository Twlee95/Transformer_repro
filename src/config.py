import torch

## Token Settings
SOS = 0
EOS = 1
PAD = 2
UNK = 3



## Random Seed
seed = 7

## Data Settings
path_ = '/workspace/Transformer/dataset9/'
max_sentence_len = 100
token_path_ = '/workspace/Transformer/NEW_Tokenizer/ende_WMT14_Tokenizer.json'
token_path_en = '/workspace/Transformer/bpe_tokenizer2/tokenizer.en.json'
token_path_de = '/workspace/Transformer/bpe_tokenizer2/tokenizer.de.json'

best_loss_model_path = './model/'
resume_checkpoint_path = None # '/workspace/Transformer/model/3200000_checkpoint.pth'

#Model Settings
MAX_LEN = 100

vocab_size = 37000
embedding_dimension = 512
hidden_dimension = 512

FeedForward_dimension = 2048 # hidden_dimension * 4 

n_head = 8
n_layers = 6
h_ff = 2048

max_tokens = 700


# Learing settings
dropout_rate = 0.1

batch_size = 40
total_steps = 100000 # batch 270000

#dataloader len 223442<- batch 20

# batch_size = 128
# total_steps = 200000

# batch_size = 256
# total_steps = 100000


label_smoothing_rate = 0.1
warmup_steps= 4000

beta_1 = 0.9
beta_2 = 0.98
eps = 1e-9


## Device Settigns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##
accumulation_step = 2


