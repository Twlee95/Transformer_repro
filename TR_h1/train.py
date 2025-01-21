from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler
import config
from load_data import prepare_data
from dataloader import TranslationDataset, collate_fn
from model import Transformer
from torch.optim.lr_scheduler import LambdaLR
import math
import torch.optim as optim
from utils import lr_lambda, load_checkpoint
import json
from tqdm import tqdm  # tqdm 라이브러리 임포트
import os

import traceback
import sys



torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"


def training_v2(model, train_dataloader, criterion, optimizer, scheduler, start_step, train_losses, train_ppls,tokenizer):
    step = start_step

    results_path = 'loss_result'
    os.makedirs(results_path, exist_ok=True)
    checkpoint_dir = config.best_loss_model_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_file = open('output_log.txt', 'w')
    print("Training is started...")

    step_threshold = 24000
    token_counts = 0
    iter_ = 0 
    
    train_loss =0.0
    # tqdm Progress Bar
    with tqdm(total=int(config.total_steps), desc="Training") as pbar:
        while step <= config.total_steps:
            model.train()

            for train_batch in train_dataloader:
                source_sentence, target_sentence, src_len, tgt_len = train_batch
                source_sentence = source_sentence.to(config.device)
                target_sentence = target_sentence.to(config.device)

                n_tokens = (sum(src_len) + sum(tgt_len)) // 2
                token_counts += n_tokens
                #tgt_sentence = target_sentence[target_sentence != 1].view(target_sentence.size(0), -1)
                
                predict = model(source_sentence, target_sentence[:, :-1])
                loss = criterion(predict.view(-1, predict.size(-1)), target_sentence[:, 1:].contiguous().view(-1))
                loss.backward()
                
                train_loss += loss.item()
                iter_ += 1

                # Gradient Accumulation Step
                if token_counts >= step_threshold:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    pbar.update(1)
                    
                    
                    # Log Perplexity and Loss
                    if (step % 20 == 0) & (step != 0):
                        train_loss = train_loss / iter_
                        train_ppl = torch.exp(torch.tensor(train_loss)).item()
    
                        log_msg = f"Step {step}: Loss = {train_loss:.4f}, Perplexity = {train_ppl:.4f}"
                        print(log_msg)
                        log_file.write(log_msg + "\n")
    
                        train_losses.append(train_loss)
                        train_ppls.append(train_ppl)

                        iter_ = 0
                        train_loss = 0.0
    
                        # Save results as JSON
                        with open(f'{results_path}/step_results.json', 'w') as f:
                            json.dump({'train_losses': train_losses, 'train_ppls': train_ppls}, f)

                    step += 1
                    token_counts = 0
                    
                # Save Checkpoint
                if (step % 10000 == 0 or (step == config.total_steps) ) and step != 0:  # or ((step) in [98500, 97000, 95500, 94000])
                    checkpoint_path = os.path.join(checkpoint_dir, f'{step}_checkpoint.pth')
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_losses': train_losses,
                        'train_ppls': train_ppls,
                        'tokenizer': tokenizer
                    }, checkpoint_path)
                    print(f"Checkpoint saved at step {step}")


                if step >= config.total_steps:
                    break

    log_file.close()


def main():
    
    # Data load
    train_pairs, tokenizer = prepare_data(config.path_ , config.token_path_, 'train',config.max_sentence_len)
    print("source data is loaded...")
    
    # Define Dataset object
    train_dataset = TranslationDataset(train_pairs,config.max_sentence_len)
    print("torch dataset is constructed...")



    model = Transformer(config.vocab_size, config.embedding_dimension,config.hidden_dimension,config.n_head,config.n_layers, config.h_ff, config.dropout_rate, config.device)
    model.to(config.device)
    print("Model is initialized...")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(),lr=1, betas=(config.beta_1,config.beta_2), eps=config.eps)
    # LambdaLR Scheduler & Label Smoothig Cross Entropy Loss
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing = config.label_smoothing_rate) #, ignore_index = config.PAD)

    
    # Define DataLoader
    # train_seq_sampler = DynamicBatchSampler(train_dataset, config.max_tokens)
    
    # Checkpoint load
    start_step = 0
    train_losses, train_ppls = [], []
    checkpoint_path = config.resume_checkpoint_path
    if checkpoint_path is not None:
        start_step, train_losses, train_ppls = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        print("Check point is loaded...")

    #train_batch_sp = BatchSampler(train_seq_sampler,batch_size = config.batch_size, drop_last=False)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, collate_fn=collate_fn)
    
    print("DataLoader is constructed...")

    # training(model, train_data_loader, criterion, optimizer, scheduler,start_step,train_losses, train_ppls)
    training_v2(model, train_data_loader, criterion, optimizer, scheduler,start_step,train_losses, train_ppls,tokenizer)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 에러 메시지를 파일에 기록
        error_log_path = 'error_log.txt'
        with open(error_log_path, 'w') as f:
            f.write("An error occurred:\n")
            f.write(traceback.format_exc())  # 전체 에러 스택 트레이스를 저장
        print(f"An error occurred. Check the log file: {error_log_path}")