import torch
from model import Transformer
import config 
from load_data import prepare_data

# ckpt.pth 파일 경로
ckpt_path = "/home/user15/TT4/model/100000_checkpoint_h8.pth"

# 파일 불러오기
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

# 내용 확인
for key in checkpoint.keys():
    print(f"{key}: {type(checkpoint[key])}")

model = Transformer(config.vocab_size, config.embedding_dimension,config.hidden_dimension,config.n_head,config.n_layers,config.h_ff, config.dropout_rate, config.device)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(config.device)

tokenizer = checkpoint['tokenizer']

# Data load
dev_pairs, tokenizer = prepare_data(config.path_ , tokenizer, 'test', config.max_sentence_len)
print("source data is loaded...")

print(len(dev_pairs[0]))

from tokenizers import Tokenizer
import math
#tokenizer = Tokenizer.from_file(config.token_path_)

output_file = "gen_sentences_h8.txt"
ref_file = "ref_sentences_h8.txt"

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
