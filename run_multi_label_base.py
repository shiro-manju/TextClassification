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

from transformers import BertJapaneseTokenizer, ElectraForSequenceClassification
model_path = "models/vanila/electra-base-japanese-discriminator"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
# electra_model = ElectraForSequenceClassification.from_pretrained(model_path, problem_type="multi_label_classification", num_labels=label_length)
electra_model = ElectraForSequenceClassification.from_pretrained(model_path, num_labels=label_length)

# freeze layers except last layer
for param in electra_model.parameters():
    param.requires_grad = False

last_layer = list(electra_model.children())[-1]
print(f'except last layer: {last_layer}')
for param in last_layer.parameters():
    param.requires_grad = True

sigmoid_func = torch.nn.Sigmoid()
# loss_func = torch.nn.BCELoss(reduction='sum')
loss_func = torch.nn.BCELoss(reduction='mean')
# loss_func = torch.nn.BCEWithLogitsLoss()
EVAL_FUNC = True # if using "Exact Match" then "True", else using "Parts Match" then "False"

LOSS_FUNC_CUSTOM = False

def train(model, device, train_loader, optimizer):
    model.train()
    running_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if LOSS_FUNC_CUSTOM:
            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            loss = loss_func(sigmoid_func(logits), labels.float())
        else:
            outputs = model(input_ids, attention_mask = attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate training loss value
    train_loss = running_loss / len(train_loader)
        
    
    model.eval()
    probs_list = []
    labels_list = []
    acc_parts_list = []
    acc_exact_list = []
    total = 0
    for idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if LOSS_FUNC_CUSTOM:
            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            loss = loss_func(sigmoid_func(logits), labels.float())
        else:
            outputs = model(input_ids, attention_mask = attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
    
        probs = sigmoid_func(logits).to('cpu')
        probs_list.append(probs)
        labels_list.append(labels.to('cpu'))

        # The label with the highest value will be our prediction 
        #_, predicted = torch.max(logits, 1) 

        total += logits.size(0) * logits.size(1)
        if idx == 15:
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

    print(f"Training: loss: {train_loss}, max_acc_exact : {max_acc_exact} then th=0.{max_acc_exact_index+1}, max_acc_parts : {max_acc_parts} then th=0.{max_acc_parts_index+1}")
    return train_loss, acc_exact_list, acc_parts_list
        
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
            labels = batch['labels'].to(device)

            if LOSS_FUNC_CUSTOM:
                outputs = model(input_ids, attention_mask = attention_mask)
                logits = outputs.logits
                loss = loss_func(sigmoid_func(logits), labels.float())
            else:
                outputs = model(input_ids, attention_mask = attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss

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

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
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


def saveModel(model): 
    path = "models/electra/npl-68_multilabel_bceloss.pth" 
    torch.save(model.state_dict(), path)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = True)

    optimizer = optim.AdamW(electra_model.parameters(), lr=2e-4)

    epoch = 5
    profile = {
        "train_loss": [],
        "train_acc_exact": [],
        "train_acc_parts": [],
        "test_loss": [],
        "test_acc_exact": [],
        "test_acc_parts": []
    }
    
    best_accuracy = -1
    electra_model.to(device)
    for i in range(epoch):
        train_loss, train_acc_exact, train_acc_parts = train(electra_model, device, train_loader, optimizer)
        profile["train_loss"].append(train_loss)
        profile["train_acc_exact"].append(train_acc_exact)
        profile["train_acc_parts"].append(train_acc_parts)
        test_loss, test_acc_exact, test_acc_parts = test(electra_model, device, test_loader)
        profile["test_loss"].append(test_loss)
        profile["test_acc_exact"].append(test_acc_exact)
        profile["test_acc_parts"].append(test_acc_parts)
        
        accuracy = max(test_acc_exact)
        if accuracy > best_accuracy: 
            saveModel(electra_model)
            best_accuracy = accuracy
    
    output_df = pd.DataFrame.from_dict(profile, orient='index').T
    output_df.to_pickle("result/profile_vanilla_oversample.pkl")
    print("best_accuracy: ", best_accuracy)
    print("End")

if __name__==('__main__'):
    main()