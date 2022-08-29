from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertJapaneseTokenizer, ElectraModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = "models/vanila/electra-base-japanese-discriminator"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
electra_model = ElectraModel.from_pretrained(model_path)

def embedding(text_list:List, max_len=512):
    
    return tokenizer(text_list, return_tensors="pt",padding=True, truncation=True, max_length=max_len)

class ElectraClassificationHead(nn.Module):
    def __init__(self, label_length):
        super(ElectraClassificationHead, self).__init__()

        # headにMLP classifierを追加
        self.activation_func =  F.gelu
        # 第1層
        self.fc1 = nn.Linear(3072, 1536) # 768*4 -> 768*2
        self.dropout1 = nn.Dropout(0.1)
        # 第2層
        self.fc2 = nn.Linear(1536, 1536)
        self.dropout2 = nn.Dropout(0.1)
        # 第3層
        self.fc3 = nn.Linear(1536, 1536)
        self.dropout3 = nn.Dropout(0.1)
        # 第4層
        self.fc4 = nn.Linear(1536, 768)
        self.dropout4 = nn.Dropout(0.1)
        # 第5層
        self.fc5 = nn.Linear(768, label_length)

        # 重み初期化処理
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.normal_(fc.weight, std=0.02)
            nn.init.normal_(fc.bias, 0)

    def forward(self, x):
        x = self.activation_func(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation_func(self.fc2(x))
        x = self.dropout2(x)
        x = self.activation_func(self.fc3(x))
        x = self.dropout3(x)
        x = self.activation_func(self.fc4(x))
        x = self.dropout4(x)
        output = self.fc5(x)
        return output

class MultilabelClassifier(nn.Module):
    def __init__(self, label_length):
        super(MultilabelClassifier, self).__init__()
        # Pre-train electra model
        self.electra_model = electra_model

        # Custom Classification Head
        self.classification_head = ElectraClassificationHead(label_length)

    def forward(self, input_ids, attention_mask):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        '''
        # 順伝搬させる
        outputs = self.electra_model(input_ids, attention_mask = attention_mask, output_hidden_states=True)
        # x = outputs.hidden_states[-1][:,0,:] # last_hidden_statesの[CLS]を抽出　-> [batch_size, embedding_ndim] 
        x = torch.cat([outputs.hidden_states[-1*i][:,0] for i in range(1, 4+1)], dim=1) # [CLS]を最小層から4層分をconcatenate
        output = self.classification_head(x)
        return output


if __name__ == ('__main__'):
    net = MultilabelClassifier(8)
    print(net)