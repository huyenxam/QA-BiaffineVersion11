import argparse
from tabnanny import check
import torch
import json
from transformers import AutoTokenizer
from model import BiaffineNER
import argparse


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BiaffineNER(args=args)
        self.model.to(self.device)
        checkpoint = torch.load(args.checkpoint_path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.max_seq_length = args.max_seq_length
        self.max_char_len = args.max_char_len

        with open(args.char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        with open(args.label_set_path, 'r', encoding='utf8') as f:
            self.label_set = f.read().splitlines()
    

    def character2id(self, character_sentence, max_seq_length):
        char_ids = []
        for word in character_sentence:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab['UNK'])
                else:
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_length:
            char_ids += [[self.char_vocab['PAD']] *  self.max_char_len] * (max_seq_length-len(char_ids))
        else:
            char_ids = char_ids[:max_seq_length]
        return torch.tensor(char_ids)
    
    def preprocess(self, tokenizer, context, question, max_seq_length, mask_padding_with_zero=True):
        firstSWindices = [0]
        input_ids = [tokenizer.cls_token_id]                    # Thêm [CLS] vào đầu câu
        firstSWindices.append(len(input_ids))

        for w in question:
            word_token = tokenizer.encode(w)                    # Chuyển các token thành số
            input_ids += word_token[1: (len(word_token) - 1)]   # Chỉ lấy token đầu tiên
                                                                # Example: seq = "Chúng tôi"
                                                                # tokenizer.encode("Chúng tôi") -> [0, 746, 2]
                                                                # Lấy token đầu tiên tại vị trí [1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))               # lưu lại vị trí token đã lấy 
        
        input_ids.append(tokenizer.sep_token_id)                # Thêm [SEP] và giữa question và context
        firstSWindices.append(len(input_ids))

        for w in context:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            if len(input_ids) >= max_seq_length:                
              firstSWindices.append(0)
            else:
              firstSWindices.append(len(input_ids))
              
        firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
        input_ids.append(tokenizer.sep_token_id)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        if len(input_ids) > max_seq_length:             
            input_ids = input_ids[:max_seq_length]
            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))
            firstSWindices = firstSWindices[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (max_seq_length - len(input_ids))
            input_ids = (
                    input_ids
                    + [
                        tokenizer.pad_token_id,
                    ]
                    * (max_seq_length - len(input_ids))
            )

            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)


    def get_character(self, word, max_char_len):
        word_rep = []
        for j in range(max_char_len):
            try:
                char = word[j]
            except:
                char = 'PAD'
            word_rep.append(char)
        return word_rep

    def get_pred_entity(self, cate_pred, span_scores):
        top_span = []
        for i in range(len(cate_pred)):
            for j in range(i, len(cate_pred)):
                if cate_pred[i][j] > 0:
                    tmp = (self.label_set[cate_pred[i][j].item()], i, j, input_tensor[i][j].item())
                    top_span.append(tmp)
        
        top_span = sorted(top_span, reverse=True, key=lambda x: x[3])

        if not top_span:
            top_span = [('ANSWER', 0, 0)]
        
        return top_span
        
    
    def predict(self, context, question):
        context = context.split(' ')
        question = question.split(' ')
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, self.max_seq_length)

        sent = question + context
        char_seq = []
        for word in sent:
            character = self.get_character(word, self.max_char_len)
            char_seq.append(character)
        if self.args.use_char:
            char_ids = self.character2id(char_seq, max_seq_length=self.max_seq_length)

        inputs = {'input_ids' :  input_ids.unsqueeze(dim=0).to(self.device),
                  'attention_mask' :  attention_mask.unsqueeze(dim=0).to(self.device),
                  'first_subword' :  firstSWindices.unsqueeze(dim=0).to(self.device),
                  'char_ids' :  char_ids.unsqueeze(dim=0).to(self.device),
        }

        with torch.no_grad():
            outputs = self.model(**input_ids)
        
        input_tensor, cate_pred = outputs[0].max(dim=-1)
        label = self.get_pred_entity(cate_pred, input_tensor)
        start_ans = label[1]
        end_ans = label[2]

        print(start_ans)
        print(end_ans)
        text_ans = " ".join(sent[start_ans:end_ans+1])
        print(text_ans)
        return text_ans

        

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--char_vocab_path', default='/content/gdrive/MyDrive/QA-BiaffineVersion11/data/charindex.json', type=str)
    parser.add_argument('--label_set_path', default='/content/gdrive/MyDrive/QA-BiaffineVersion11/data/label_set.txt', type=str)
    parser.add_argument('--model_name_or_path', default='xlm-roberta-large', type=str)
    parser.add_argument('--max_char_len', default=10, type=int)
    parser.add_argument('--max_seq_length', default=400, type=int)

    # model
    parser.add_argument('--use_char', action="store_true")
    parser.add_argument('--char_embedding_dim', default=100, type=int)
    parser.add_argument('--char_hidden_dim', default=200, type=int)
    parser.add_argument('--num_layer_bert', default=1, type=int)
    parser.add_argument('--char_vocab_size', default=108, type=int)
    parser.add_argument('--hidden_dim', default=728, type=int)
    parser.add_argument('--hidden_dim_ffw', default=400, type=int)
    parser.add_argument('--num_labels', default=12, type=int)

    parser.add_argument('--checkpoint_path', default='/content/gdrive/MyDrive/QA-BiaffineVersion11/results/checkpoint.pth')
    args, unk = parser.parse_known_args()

    
    question = "Bình được công nhận với danh hiệu gì ?"
    context = "Bình Nguyễn là một người đam mê với lĩnh vực xử lý ngôn ngữ tự nhiên . Anh nhận chứng chỉ Google Developer Expert năm 2020"
    
    p = Predictor(args)
    print(p.predict(context, question))

