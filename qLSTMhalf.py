import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import jieba
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from torch import optim
from torch.nn import functional as F
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
# import qiskit_algorithms
import torch
# Importing Numpy
import numpy as np
from numpy.random import default_rng
import pandas as pd
# from IPython.display import clear_output
import pickle
import torch
import torch.nn.functional as F
# from qiskit import Aer

import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
'''定义一个词表类型。'''

# 该类用于实现token到索引的映射
class Vocab:

    def __init__(self, tokens=None) -> None:
        # 构造函数
        # tokens：全部的token列表

        self.idx_to_token = list()
        # 将token存成列表，索引直接查找对应的token即可
        self.token_to_idx = dict()
        # 将索引到token的映射关系存成字典，键为索引，值为对应的token

        if tokens is not None:
            # 构造时输入了token的列表
            if "<unk>" not in tokens:
                # 不存在标记
                tokens = tokens + "<unk>"
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                # 当前该token对应的索引是当下列表的最后一个
            self.unk = self.token_to_idx["<unk>"]

    @classmethod

    def build(cls, data, min_freq=1, reserved_tokens=None, stop_words='stopwords_cn.txt'):
        # 构建词表
        # cls：该类本身
        # data: 输入的文本数据
        # min_freq：列入token的最小频率
        # reserved_tokens：额外的标记token
        # stop_words：停用词文件路径
        token_freqs = defaultdict(int)
        with open(stop_words, encoding='utf-8') as f:
            stopwords = f.read().split('\n')
        for i in tqdm(range(data.shape[0]), desc=f"Building vocab"):
            for token in jieba.lcut(data.iloc[i]["review"]):
                if token in stopwords:
                    continue
                token_freqs[token] += 1
        # 统计各个token的频率
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        # 加入额外的token
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        # 全部的token列表
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词表的大小
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入token对应的索引，不存在则返回<unk>返回的索引
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标签对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in ids]


'''数据集构建函数'''


def build_data(data_path: str):
    '''
    Args:
       data_path:待读取本地数据的路径
    Returns:
       训练集、测试集、词表
    '''
    whole_data = pd.read_csv(data_path)
    # 读取数据为 DataFrame 类型
    vocab = Vocab.build(whole_data)
    # 构建词表

    train_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in
                  whole_data[whole_data["label"] == 1][:50000]["review"]] \
                 + [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in
                    whole_data[whole_data["label"] == 0][:50000]["review"]]
    # 分别取褒贬各50000句作为训练数据，将token映射为对应的索引值

    test_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in
                 whole_data[whole_data["label"] == 1][50000:]["review"]] \
                + [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in
                   whole_data[whole_data["label"] == 0][50000:]["review"]]
    # 其余数据作为测试数据
    print(len(vocab))
    return train_data, test_data, vocab


'''声明一个 DataSet 类'''


class MyDataset(Dataset):

    def __init__(self, data) -> None:
        # data：使用词表映射之后的数据
        self.data = data

    def __len__(self):
        # 返回样例的数目
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


'''声明一个collate_fn函数，用于对一个批次的样本进行整理'''


def collate_fn(examples):
    # 从独立样本集合中构建各批次的输入输出
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    # 获取每个序列的长度
    inputs = [torch.tensor(ex[0]) for ex in examples]

    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 目标targets为该批次所有样例输出结果构成的张量
    inputs = pad_sequence(inputs, batch_first=True)
    # 将用pad_sequence对批次类的样本进行补齐
    return inputs, lengths, targets

# 创建一个QuantumLSTM类作为模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

class QuantumLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(QuantumLayer, self).__init__()

        # 这里只是一个示例，实际应用中需要根据具体的量子电路设计来定义
        self.input_size = input_size
        self.output_size = output_size
        self.quantum_op = nn.Linear(input_size, output_size)  # 示例的量子操作

    def forward(self, x):
        # 在前向传播中执行量子电路操作
        return self.quantum_op(x)

class LSTMWithQuantum(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, quantum_input_size, quantum_output_size):
        super(LSTMWithQuantum, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.quantum_layer = QuantumLayer(quantum_input_size, quantum_output_size)  # 添加量子层
        self.output = nn.Linear(hidden_dim + quantum_output_size, num_class)  # 将量子层的输出和LSTM的隐藏状态连接到输出层

    def forward(self, inputs, lengths):
        embeds = self.embedding(inputs)
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(x_pack)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # 提取LSTM最后一个时间步的输出作为输入
        lstm_output_last = lstm_output[torch.arange(lstm_output.size(0)), lengths - 1, :]
        # 将LSTM的输出和量子层的输出拼接起来
        combined_output = torch.cat((lstm_output_last, self.quantum_layer(lstm_output_last)), dim=1)
        # 经过全连接层得到最终输出
        output = self.output(combined_output)
        return F.log_softmax(output, dim=-1)

'''训练'''

embedding_dim = 128
hidden_dim = 24
batch_size = 1024
num_epoch = 100
num_class = 2
quantum_input_size = hidden_dim  # 示例中假设量子层输入维度与LSTM隐藏层维度相同
quantum_output_size = 32  # 示例中假设量子层输出维度为32

train_data, test_data, vocab = build_data("smallsampled_file.csv")
# 加载数据
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
print(len(test_data_loader))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMWithQuantum(len(vocab), embedding_dim, hidden_dim, num_class,quantum_input_size,quantum_output_size)
model.to(device)
# 加载模型

nll_loss = nn.NLLLoss()
# 负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Adam优化器

model.train()
from sklearn.metrics import precision_recall_fscore_support
import time
import csv
from sklearn.metrics import precision_recall_fscore_support

# 定义用于记录损失和准确率的列表
train_losses = []
train_accuracies = []
train_recalls = []
train_f1_scores = []  # 添加用于记录F1值的列表

# 训练循环
for epoch in range(num_epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}"):
        inputs, lengths, targets = [x.to(device) for x in batch]

        # 正向传播
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total_correct += correct
        total_samples += targets.size(0)
        # 正确处理预测值，假设二分类问题
        probabilities = torch.sigmoid(outputs)  # 应用sigmoid激活函数得到概率
        _, predicted_classes = torch.max(probabilities, 1)  # 根据概率预测类别
        # 反向传播计算梯度
        loss.backward()


        optimizer.step()

        total_loss += loss.item()

    # 保存每个 epoch 的平均损失和准确率
    avg_loss = total_loss / len(train_data_loader)
    train_losses.append(avg_loss)
    train_accuracy = total_correct / total_samples
    train_accuracies.append(train_accuracy)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_data_loader)
    train_accuracy = total_correct / total_samples
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    # 计算召回率和F1分数，需要转换为numpy数组
    predictions = predicted_classes.cpu().numpy()
    targets = targets.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')

    train_recalls.append(recall)
    train_f1_scores.append(f1)

    print(
        f"Epoch {epoch + 1}/{num_epoch}, Average Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f}")


print(f"Epoch {epoch + 1}/{num_epoch}, Average Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f}")

# 记录结束训练的时间点
end_time = time.time()

# 输出到CSV文件
with open('train_final2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Average Loss', 'Train Accuracy', 'Train Recall', 'Train F1'])
    for epoch in range(num_epoch):
        writer.writerow(
            [epoch + 1, train_losses[epoch], train_accuracies[epoch], train_recalls[epoch], train_f1_scores[epoch]])