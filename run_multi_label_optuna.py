import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import torch.nn.functional as F
import torch.optim as optim
import pickle
import csv

print(torch.__version__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from transformers import BertJapaneseTokenizer, ElectraForPreTraining
model_path = "models/vanila/electra-base-japanese-discriminator"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
electra_model = ElectraForPreTraining.from_pretrained(model_path)
for param in electra_model.parameters():
    param.requires_grad = False
electra_model.to(device)

sigmoid_func = torch.nn.Sigmoid()
loss_func = torch.nn.BCELoss()
# loss_func = torch.nn.BCEWithLogitsLoss()
EVAL_FUNC = True # if using "Exact Match" then "True", else using "Parts Match" then "False"

label2idx={
    0: '移行方式設計',
    '移行方式設計': 0,
    1: '基本設計',
    '基本設計': 1,
    2: '基盤方式設計',
    '基盤方式設計': 2,
    3: '内部・詳細設計',
    '内部・詳細設計': 3,
    4: '切替方式設計',
    '切替方式設計': 4,
    5: '運用設計',
    '運用設計': 5,
    6: '標準化',
    '標準化': 6,
    7: '概要設計',
    '概要設計': 7
 }
label_length = 8

def get_optimizer(trial, model):
    optimizer_names = ['AdamW', 'Adam'] # 'rmsprop'
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)


    if optimizer_name == optimizer_names[0]: 
        adamw_lr = trial.suggest_loguniform('adamw_lr', 5e-4, 5e-3)
        optimizer = optim.AdamW(model.parameters(), lr=adamw_lr)
    if optimizer_name == optimizer_names[1]: 
        adam_lr = trial.suggest_loguniform('adam_lr', 5e-4, 5e-3)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr)
    else:
        adadelta_lr = trial.suggest_loguniform('adadelta_lr', 5e-4, 5e-3)
        optimizer = optim.Adadelta(model.parameters(), lr=adadelta_lr)

    return optimizer

def get_activation(trial):
    activation_names = ['Softplus', 'GeLU']
    activation_name = trial.suggest_categorical('activation', activation_names)
    
    if activation_name == activation_names[0]:
        activation = F.softplus
    # elif activation_name == activation_names[1]:
    #     activation = F.gelu
    else:
        activation = F.gelu
    return activation

def train(model, device, train_loader, optimizer):
    model.train()
    running_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = electra_model(input_ids, attention_mask = attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1][:,0,:]
        logits = model(last_hidden_states)
        loss = loss_func(sigmoid_func(logits), labels.float())
        
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    model.eval()
    probs_list = []
    labels_list = []
    acc_parts_list = []
    acc_exact_list = []
    total = 0
    for idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        outputs = electra_model(input_ids, attention_mask = attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1][:,0,:]
        logits = model(last_hidden_states)
    
        probs = sigmoid_func(logits).to('cpu')
        probs_list.append(probs)
        labels_list.append(labels.to('cpu'))

        # The label with the highest value will be our prediction 
        #_, predicted = torch.max(logits, 1) 

        total += logits.size(0) * logits.size(1)
        if idx == 20:
            break

    probs_ = torch.cat(probs_list, 0)
    labels_ = torch.cat(labels_list, 0)   
    for th in range(1, 10, 1):
        correct_1 = 0
        correct_2 = 0
        predicted = torch.where(probs_ > th/10, torch.ones(len(labels_), label_length), torch.zeros(len(labels_), label_length))
        correct_1 += (predicted == labels_).all(axis=1).sum().item()
        acc_exact_list.append(correct_1 / len(labels_))
        correct_2 += (predicted == labels_).sum().item()
        acc_parts_list.append(correct_2 / total)

    max_acc_exact = max(acc_exact_list)
    max_acc_exact_index = acc_exact_list.index(max_acc_exact)
    max_acc_parts = max(acc_parts_list)
    max_acc_parts_index = acc_parts_list.index(max_acc_parts)

    print(f"Training: loss: {running_loss/len(train_loader)}, max_acc_exact : {max_acc_exact} then th=0.{max_acc_exact_index+1}, max_acc_parts : {max_acc_parts} then th=0.{max_acc_parts_index+1}")
    return running_loss/len(train_loader), acc_exact_list, acc_parts_list
        
def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        acc_exact_list = []
        acc_parts_list = []
        running_loss = 0
        total = 0
        probs_list = []
        labels_list = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = electra_model(input_ids, attention_mask = attention_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][:,0,:]
            logits = model(last_hidden_states)
            loss = loss_func(sigmoid_func(logits).to('cpu'), labels.float())

            probs = sigmoid_func(logits).to('cpu')
            probs_list.append(probs)
            labels_list.append(labels.to('cpu'))

            # The label with the highest value will be our prediction 
            #_, predicted = torch.max(logits, 1) 

            total += logits.size(0) * logits.size(1)
            running_loss += loss.item()

        probs_ = torch.cat(probs_list, 0)
        labels_ = torch.cat(labels_list, 0)   
        for th in range(1, 10, 1):
            correct_1 = 0
            correct_2 = 0
            predicted = torch.where(probs_ > th/10, torch.ones(len(labels_), label_length), torch.zeros(len(labels_), label_length))
            correct_1 += (predicted == labels_).all(axis=1).sum().item()
            acc_exact_list.append(correct_1 / len(labels_))
            correct_2 += (predicted == labels_).sum().item()
            acc_parts_list.append(correct_2 / total)

    max_acc_exact = max(acc_exact_list)
    max_acc_exact_index = acc_exact_list.index(max_acc_exact)
    max_acc_parts = max(acc_parts_list)
    max_acc_parts_index = acc_parts_list.index(max_acc_parts)

    print(f"Eval: loss: {running_loss/len(test_loader)}, max_acc_exact : {max_acc_exact} then th=0.{max_acc_exact_index+1}, max_acc_parts : {max_acc_parts} then th=0.{max_acc_parts_index+1}")
    return running_loss/len(test_loader), acc_exact_list, acc_parts_list

class MLP(nn.Module):
    def __init__(self, trial):
        super().__init__()
        self.activation_func = get_activation(trial) # relu sigmoid gelu
        # 第1層
        self.fc1 = nn.Linear(768, 600) # 1152 , 3072, 
        self.dropout1 = nn.Dropout(0.1)
        # 第2層
        self.fc2 = nn.Linear(600, 600) #768 -> 600
        self.dropout2 = nn.Dropout(0.1)
        # 第3層
        self.fc3 = nn.Linear(600, 600) #768 -> 600
        self.dropout3 = nn.Dropout(0.1)
        # 第4層
        self.fc4 = nn.Linear(600, 300)
        self.dropout4 = nn.Dropout(0.1)
        # 第5層
        self.fc5 = nn.Linear(300, label_length)
 
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

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key:torch.tensor(val[idx]).clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32).clone().detach()
        return item
    
    def __len__( self):
        return len(self.labels)

def load_df():
    csv.field_size_limit(1000000000)
    content_list = []
    label_list = []
    with open("NPL-69/npl-69_dataset_for_multilabel_multiclass_classification.csv", encoding="utf-8", errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # content
            content = row[3]
            if len(content) > 8000:
                content = content[:8000]
            content_list.append(content)

            # label
            labels = row[2]
            label_vec = [0]*(label_length)
            labels = labels.split(",")
            for label in labels:
                label_vec[label2idx[label]] = 1
            label_list.append(label_vec)
            
            # if len(label_list) > 1000:
            #     break

    df = pd.DataFrame({"content":content_list, "label":label_list})
    del content_list
    del label_list
    return df

def main():
    print("Data Loading ...")
    df = load_df()
    train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True)
    # train_df = pd.read_pickle("NPL-69/npl-69_multiclass_oversample_train.pkl")
    # test_df = pd.read_pickle("NPL-69/npl-69_multiclass_oversample_test.pkl")

    max_len = 512
    train_encoding = tokenizer(train_df["content"].to_list(), return_tensors="pt",padding=True, truncation=True, max_length=max_len)
    test_encoding = tokenizer(test_df["content"].to_list(), return_tensors="pt",padding=True, truncation=True, max_length=max_len)

    train_label = torch.tensor(train_df["label"].to_list())
    test_label = torch.tensor(test_df["label"].to_list())

    train_dataset = CreateDataset(train_encoding, train_label)
    test_dataset = CreateDataset(test_encoding, test_label)
    del train_encoding
    del test_encoding

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = True)

    def objective(trial):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlp_model= MLP(trial).to(device)
        optimizer = get_optimizer(trial, mlp_model)

        epoc_profile = {}
        max_acc = 0
        for step in range(EPOCH):
            print(f"=== epoc: {step+1} ===")
            train_loss, train_acc_exact_list, train_acc_parts_list= train(mlp_model, device, train_loader, optimizer)
            test_loss, acc_exact_list, acc_parts_list = test(mlp_model, device, test_loader)

            epoc_profile[f"epoc{step}_test_loss"] = test_loss
            epoc_profile[f"epoc{step}_acc_exact"] = acc_exact_list
            epoc_profile[f"epoc{step}_acc_parts"] = acc_parts_list
            epoc_profile[f"epoc{step}_train_loss"] = train_loss
            epoc_profile[f"epoc{step}_train_acc_exact"] = train_acc_exact_list
            epoc_profile[f"epoc{step}_train_acc_parts"] = train_acc_parts_list

            
            if max(acc_exact_list) > max_acc:
                with open(f"models/checkpoints/trialnum_{trial.number+1}.pkl", "wb") as fout:
                    pickle.dump(mlp_model, fout)
                    max_acc = max(acc_exact_list)

        trial.set_user_attr(f'profile', epoc_profile)
        return (1 - max(acc_exact_list))

    TRIAL_SIZE = 3
    EPOCH = 10 
    print("Training start ...")
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)

    print(study.best_params)
    hist_df = study.trials_dataframe(multi_index=True)
    hist_df.to_pickle("result/profile_v2.pkl")

    print("End")

if __name__==('__main__'):
    main()