import torch
from model import Transformer
import config 
from load_data import prepare_data

# ckpt.pth 파일 경로
#ckpt_path = "/home/user15/TT4/model/100000_checkpoint_h8.pth"
ckpt_paths = ["/home/user15/TT4/model/94000_checkpoint_h8.pth", \
            "/home/user15/TT4/model/95500_checkpoint_h8.pth", \
            "/home/user15/TT4/model/97000_checkpoint_h8.pth", \
            "/home/user15/TT4/model/98500_checkpoint_h8.pth", \
            "/home/user15/TT4/model/100000_checkpoint_h8.pth"]

checkpoints = [torch.load(ckpt, map_location="cpu") for ckpt in ckpt_paths]



# 첫 번째 체크포인트를 기반으로 평균 가중치를 저장할 새로운 dict 생성
avg_model_state_dict = {key: torch.zeros_like(value) for key, value in checkpoints[0]['model_state_dict'].items()}

# 모든 모델 가중치를 더하기
for checkpoint in checkpoints:
    for key in avg_model_state_dict.keys():
        avg_model_state_dict[key] += checkpoint['model_state_dict'][key]

# 평균 계산
for key in avg_model_state_dict.keys():
    avg_model_state_dict[key] /= len(checkpoints)

# 새로운 체크포인트 딕셔너리 생성 (기존 정보 유지)
new_checkpoint = {
    'step': checkpoints[0]['step'],  # 첫 번째 체크포인트의 step 값 사용
    'model_state_dict': avg_model_state_dict,  # 평균 가중치 적용
    'optimizer_state_dict': checkpoints[0]['optimizer_state_dict'],  # 첫 번째 옵티마이저 상태 유지
    'scheduler_state_dict': checkpoints[0]['scheduler_state_dict'],  # 첫 번째 스케줄러 상태 유지
    'train_losses': checkpoints[0]['train_losses'],  # 첫 번째 loss 정보 유지
    'train_ppls': checkpoints[0]['train_ppls'],  # 첫 번째 perplexity 정보 유지
    'tokenizer': checkpoints[0]['tokenizer'],  # 첫 번째 tokenizer 유지
}


model = Transformer(config.vocab_size, config.embedding_dimension,config.hidden_dimension,config.n_head,config.n_layers,config.h_ff, config.dropout_rate, config.device)
model.load_state_dict(new_checkpoint['model_state_dict'], strict=True)
model.to(config.device)

tokenizer = new_checkpoint['tokenizer']

# Data load
dev_pairs, tokenizer = prepare_data(config.path_ , tokenizer, 'dev', config.max_sentence_len)
print("source data is loaded...")

print(len(dev_pairs[0]))

from tokenizers import Tokenizer
import math
#tokenizer = Tokenizer.from_file(config.token_path_)

output_file = "gen_sentences_avg.txt"
ref_file = "ref_sentences_avg.txt"

# 파일 초기화 (기존 파일 삭제 후 새로 생성)
with open(output_file, "w", encoding="utf-8") as f1, open(ref_file, "w", encoding="utf-8") as f2:

    model.eval()
    with torch.no_grad():
        for src, tgt in dev_pairs:
            input_src = torch.tensor(src, dtype=torch.long).to(config.device)
            output_tgt = torch.tensor(tgt, dtype=torch.long).to(config.device)

            #print(input_src.size())
            # Encoder 처리
            enc_attn_mask = model.gen_att_mask(input_src.unsqueeze(0), config.PAD)  # batch, seq
            # print("enc_attn_mask",enc_attn_mask.size())
            # print("enc_attn_mask",enc_attn_mask)
            embedded_src = model.embedding(input_src.unsqueeze(0)) * math.sqrt(config.embedding_dimension)
            pos_src = model.Positional_Embedding(embedded_src, config.device).unsqueeze(0)
            dr_src = model.dropout(pos_src + embedded_src)
            encoder_output = model.transformer_encoder(dr_src, enc_attn_mask)

            # print("encoder_output",encoder_output.size())
            # print("encoder_output",encoder_output)

            # Decoding 시작
            start_tk = torch.tensor([tokenizer.encode("<s>").ids[0]], dtype=torch.long).to(config.device)  # <s> token

            # print("start_tk",start_tk.unsqueeze(0))
            decoder_input = start_tk.unsqueeze(0)  # 첫 입력은 시작 토큰

            sequences = [[[start_tk.item()], 0.0, decoder_input]] # (sequence, score, decoder input)

            for _ in range(config.MAX_LEN):
                all_candidates = []

                for seq, score, decoder_input in sequences:
                    # 종료 토큰 <eos> 확인
                    if len(seq) > 0 and seq[-1] == tokenizer.encode("</s>").ids[0]:
                        all_candidates.append([seq, score, decoder_input])
                        continue

                    #print("decoder_input: ", decoder_input.size())
                    dec_attn_mask = model.gen_att_mask(decoder_input, config.PAD)
                    # 현재 입력 처리
                    embedded_st = model.embedding(decoder_input) * math.sqrt(config.embedding_dimension)
                    pos_st = model.Positional_Embedding(embedded_st, config.device)
                    dec_embed_input = model.dropout(pos_st + embedded_st)

                    # 디코더 실행
                    decoder_output = model.transformer_decoder(encoder_output, dec_embed_input, enc_attn_mask, dec_attn_mask)
                    
                    # print("decoder_output",decoder_output.size())
                    # print("decoder_output",decoder_output[:, -1, :].size())
                    #print("decoder_output",decoder_output[:, -1, :])

                    logits = model.fc_out(decoder_output[:, -1, :])  # 마지막 타임스텝의 출력
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    # Top-k 추출
                    topk_log_prob, topk_ids = torch.topk(log_probs, 4, dim=-1)

                    for j in range(4):  # beam size
                        word_idx = topk_ids[0, j].item()  # 새로 생성된 단어의 인덱스
                        prob = topk_log_prob[0, j].item()  # 해당 단어의 확률

                        # 새로운 시퀀스 업데이트
                        new_sequence = seq + [word_idx]  # 기존 시퀀스에 새로운 단어 추가

                        # 새로운 입력 생성 (현재 시퀀스에 따라 업데이트)
                        new_decoder_input = torch.tensor([new_sequence], dtype=torch.long, device=config.device)
                        #new_decoder_input = torch.tensor([[word_idx]], dtype=torch.long, device=config.device)
                        #print("new_decoder_input: ", new_decoder_input.size())

                        # 점수 업데이트
                        new_score = score + prob

                        # 새로운 후보 추가
                        all_candidates.append([new_sequence, new_score, new_decoder_input])

                # Top-k 유지
                #print("all_candidates",all_candidates)
                ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                sequences = ordered[:4]  # beam size 유지

                # 모든 시퀀스가 종료되었는지 확인
                if all(len(seq) > 0 and seq[-1] == tokenizer.encode("</s>").ids[0] for seq, _, _ in sequences):
                    break

            # 최종 시퀀스 선택
            
            best_sequence = sequences[0][0]

            #print("sequences",sequences)
            # 종료 토큰 제거
            if best_sequence[-1] == tokenizer.encode("</s>").ids[0]:
                best_sequence = best_sequence[1:-1]

            # print("best_sequence",best_sequence)
            
            # 디코딩
            
            target_sentence = tokenizer.decode(output_tgt.tolist())
            generated_sentence = tokenizer.decode(best_sequence)
            source_sentence = tokenizer.decode(input_src.tolist())

            # 결과 출력
            #print(f"input_src: {input_src.tolist()}")
            print(f"Source: {source_sentence}")

            #print(f"best_sequence: {best_sequence}")
            print(f"Generated: {generated_sentence}")

            #print("output_tgt",output_tgt.tolist())
            print(f"Target: {target_sentence}")
            print("@@@@@@@@@@@@@")

            # 파일 저장
            f1.write(generated_sentence + "\n")
            f2.write(target_sentence + "\n")
