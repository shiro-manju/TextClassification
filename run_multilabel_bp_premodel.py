import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from src.util import CreateDataset, load_df, demo_load_df
from src.ClassificationHeader import MultilabelClassifier, embedding

print(torch.__version__)

sigmoid_func = torch.nn.Sigmoid()
loss_func = torch.nn.BCELoss(reduction='mean')
EVAL_FUNC = True # if using "Exact Match" then "True", else using "Parts Match" then "False"

def train(model, device, label_length, train_loader, optimizer):
    model.train()
    running_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_func(sigmoid_func(logits), labels.float())

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
        
        logits = model(input_ids, attention_mask)
        loss = loss_func(sigmoid_func(logits), labels.float())

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
        
def test(model, device, label_length, test_loader):
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

            logits = model(input_ids, attention_mask = attention_mask)
            loss = loss_func(sigmoid_func(logits), labels.float())

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


def saveModel(model): 
    path = "models/electra/multilabel_crassification_bp_pretrain.pth" 
    torch.save(model.state_dict(), path)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Data Loading ...")
    label2idx, label_length, df = demo_load_df()
    print(df.head())
    
    train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True)
    # train_df = pd.read_pickle("NPL-69/npl-69_multiclass_oversample_train.pkl")
    # test_df = pd.read_pickle("NPL-69/npl-69_multiclass_oversample_test.pkl")

    max_len = 512
    train_encoding = embedding(train_df["content"].to_list(), max_len)
    test_encoding = embedding(test_df["content"].to_list(), max_len)

    train_label = torch.tensor(train_df["label"].to_list())
    test_label = torch.tensor(test_df["label"].to_list())

    train_dataset = CreateDataset(train_encoding, train_label)
    test_dataset = CreateDataset(test_encoding, test_label)

    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = True)
    
    
    # loading network
    net = MultilabelClassifier(label_length)
    print(net)
    # freeze backbrop
    for param in net.parameters():
        param.requires_grad = False
    for param in net.electra_model.encoder.layer[-4:].parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': net.electra_model.encoder.layer[-4].parameters(), 'lr': 1e-6},
        {'params': net.electra_model.encoder.layer[-3].parameters(), 'lr': 5e-6},
        {'params': net.electra_model.encoder.layer[-2].parameters(), 'lr': 1e-5},
        {'params': net.electra_model.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': net.classification_head.parameters(), 'lr': 1e-3}
    ])


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
    net.to(device)
    for i in range(epoch):
        train_loss, train_acc_exact, train_acc_parts = train(net, device, label_length, train_loader, optimizer)
        profile["train_loss"].append(train_loss)
        profile["train_acc_exact"].append(train_acc_exact)
        profile["train_acc_parts"].append(train_acc_parts)
        test_loss, test_acc_exact, test_acc_parts = test(net, device, label_length, test_loader)
        profile["test_loss"].append(test_loss)
        profile["test_acc_exact"].append(test_acc_exact)
        profile["test_acc_parts"].append(test_acc_parts)
        
        accuracy = max(test_acc_exact)
        if accuracy > best_accuracy: 
            saveModel(net)
            best_accuracy = accuracy
    
    output_df = pd.DataFrame.from_dict(profile, orient='index').T
    output_df.to_pickle("result/profile_last4layer_bp_used_last4cls.pkl")
    print("best_accuracy: ", best_accuracy)
    print("End")

if __name__==('__main__'):
    main()