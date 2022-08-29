import torch
import csv
import pandas as pd

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

    return label2idx, label_length, df

def demo_load_df():
    df = pd.DataFrame.from_dict(pd.read_pickle('./data/twitter/twitterJSA_data.pickle'))
    label2idx = {0:"ポジティブ&ネガティブ",
                1:"ポジティブ",
                2:"ネガティブ",
                3:"ニュートラル",
                4:"無関係"}
    label_length = 5

    # data数削減
    df = df.sample(n=7000, random_state=123)
    df = df.rename(columns={'text': 'content'})

    return label2idx, label_length, df

    # train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True)
    # train_df, valid_df = train_test_split(train_df, test_size=0.2, shuffle=True)
    # print(train_df["label"].value_counts())
    # print(test_df["label"].value_counts())