import jieba
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import time
import itertools
from tensorflow.keras.preprocessing import sequence
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os

replies_dataframe = pd.read_csv("./saved/replies_17.csv")
replies = replies_dataframe["content"]
replies_dataframe_binary = replies_dataframe["num_like"] >= 100
replies_dataframe["high_like"] = replies_dataframe_binary.astype(int)
replies_dataframe_processed = replies_dataframe[["content", "high_like"]]
# waimai_dataframe = pd.read_csv("./saved/waimai_10k.csv")
# review = waimai_dataframe["review"]
text_size = 0
vocab_dict = {}


def get_vocab_size():
    global text_size
    vocab = set()
    for reply in replies:
        parsed = jieba.lcut(reply)
        for word in parsed:
            vocab.add(word)
            if not vocab_dict.keys().__contains__(word):
                vocab_dict.update({word: len(vocab)-1})
            text_size = text_size + 1

    return len(vocab)


BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextLikes(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        """
        description: initialization function of the class
        :param vocab_size: size of vocabulary for the entire text dataset
        :param embed_dim: dimension for word embedding
        :param num_class: number of classes
        """""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, 60)
        self.fc2 = nn.Linear(60, num_class)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for the network"""
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text):
        """
        :param text: result of text being mapped to numebr
        :return: tensor of same size of number of classes for classification
        """
        embedded = self.embedding(text)
        c = embedded.size(0) // BATCH_SIZE
        embedded = embedded[:BATCH_SIZE*c]
        embedded = embedded.transpose(1, 0).unsqueeze(0)
        embedded = f.avg_pool1d(embedded, kernel_size=c)
        out = self.fc2(f.relu(self.fc1(embedded[0].transpose(1, 0))))
        return out


VOCAB_SIZE = get_vocab_size()
EMBED_DIM = 32
NUM_CLASS = 2
model = TextLikes(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
criterion = nn.CrossEntropyLoss(torch.tensor([1.0, 100.0])).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


def generate_batch(batch):
    label = torch.tensor([entry[1] for entry in batch])
    text_number = [list(map(lambda x: vocab_dict[x], jieba.lcut(entry[0]))) for entry in batch]
    text_number = sequence.pad_sequences(text_number, 200)
    text_number = list(map(lambda x: torch.tensor(x), text_number))
    text = torch.cat(text_number)
    label = label.type(torch.LongTensor)
    return text, label


def train(train_data):
    train_loss = 0
    train_acc = 0
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    high_like_predicted = 0
    for i, (text, cls) in enumerate(data):
        optimizer.zero_grad()
        output = model(text)
        if len(cls) != 16:
            break
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += output.argmax(1).eq(cls).logical_and(cls.eq(1)).sum().item()
        high_like_predicted += output.argmax(1).eq(1).sum().item()

    scheduler.step()

    high_like = 0
    for data in train_data:
        if data["high_like"] == 1:
            high_like += 1

    return train_loss / len(train_data), train_acc / high_like, train_acc / high_like_predicted


def valid(valid_data):
    valid_loss = 0
    acc = 0
    high_like_predicted = 0
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for i, (text, cls) in enumerate(data):
        if len(cls) != 16:
            break
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, cls)
            valid_loss += loss.item()
            acc += output.argmax(1).eq(cls).logical_and(cls.eq(1)).sum().item()
            high_like_predicted += output.argmax(1).eq(1).sum().item()

    high_like = 0
    for data in valid_data:
        if data["high_like"] == 1:
            high_like += 1

    return valid_loss / len(valid_data), acc / high_like, acc / high_like_predicted


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.frame = dataframe

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.get_item()

        data = self.frame.iloc[idx, :]

        return data


dataset = MyDataset(replies_dataframe_processed)

N_EPOCHS = 50
min_valid_loss = float('inf')
train_len = int(len(dataset) * 0.7)
sub_train_, sub_valid_ = \
    random_split(dataset, [train_len, len(dataset) - train_len])

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_precision, train_rec = train(sub_train_)
    valid_loss, valid_precision, valid_rec = valid(sub_valid_)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tPrecision: {train_precision * 100:.1f}%(train)\t|\tRecall: {train_rec * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tPrecision: {valid_precision * 100:.1f}%(valid)\t|\tRecall: {valid_rec * 100:.1f}%(valid)')

print(model.state_dict()['embedding.weight'])
torch.save(model.state_dict(), "./saved/model_100_17")
torch.save(vocab_dict, "./saved/vocab_dict_100_17")
