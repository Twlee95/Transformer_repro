import config
import torch
# Lambda function for learning rate
def lr_lambda(step_num):
    d_model = config.hidden_dimension
    warmup_steps = config.warmup_steps
    scale = d_model ** -0.5
    
    if step_num == 0:
        step_num = 1
    
    return scale * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)





def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step = checkpoint['step']
    train_losses = checkpoint['train_losses']
    train_ppls = checkpoint['train_ppls']


    print(f"체크포인트 {checkpoint_path}에서 step {step}부터 학습을 재개합니다.")
    return step, train_losses, train_ppls