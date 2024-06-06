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
import pandas as pd
# from IPython.display import clear_output
import torch.optim as optim
import time
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

from tqdm import tqdm


class QuantumInspiredActivation(nn.Module):
    def __init__(self, rotation_axis='x', dimension=-1):
        super().__init__()
        self.rotation_axis = rotation_axis.lower()
        self.dimension = dimension
        # 用可学习的参数代替固定旋转矩阵
        self.theta = nn.Parameter(torch.rand(1))  # 单一旋转角度参数
        self.phi = nn.Parameter(torch.rand(1))  # Additional rotation angle parameter for more complex rotations

    def forward(self, x):
        if self.rotation_axis == 'x':
            rotated_features = x * torch.cos(self.theta) + torch.sin(self.theta) * torch.zeros_like(x)
        elif self.rotation_axis == 'y':
            # 用复数模拟y轴旋转的量子操作
            x_complex = x.float().to(torch.complex64)  # Assuming x is real, converting to complex
            rotated_features = x_complex * torch.exp(1j * self.theta)  # 关于旋转的欧拉公式
            # If phi was intended for use in Y-axis rotation, you'd incorporate it here, e.g., `torch.exp(1j * (self.theta + self.phi))`
        elif self.rotation_axis == 'z':
            rotated_features = x  # Direct return or consider phase shift with phi, though phi isn't typically used for Z-axis
        else:
            raise ValueError("Unsupported rotation axis.")

        # Conversion back to real if necessary (assuming this logic is desired)
        if rotated_features.is_complex() and not torch.is_complex(x):
            rotated_features = torch.view_as_real(rotated_features).sum(dim=-1)

        return rotated_features
class SymbolicQuantumCNOT(nn.Module):
    def __init__(self):
        super(SymbolicQuantumCNOT, self).__init__()

    def forward(self, control, target):
        # 确保控制比特的状态用于生成布尔掩码
        control_mask = (control[:, 0] > 0).float()  # 假设大于0为True（激活态），等于0为False（基态）
        flipped_target = target * (-1)  # 简化的“翻转”逻辑，仅为示意
        # 现在确保control_mask是布尔类型的，用于torch.where
        control_mask = control_mask.bool()
        result_target = torch.where(control_mask.unsqueeze(-1), target, flipped_target)
        return control, result_target

# 修改量子门模块
class QuantumGate(nn.Module):#关键点量子门对应lstm里面的门
    def __init__(self, input_size):
        super(QuantumGate, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_size, input_size))#修改为可更新权重的结构

    def forward(self, x):
        return torch.matmul(x, self.weight)

class QuantumInspiredOutputGate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuantumInspiredOutputGate, self).__init__()
        self.linear_transform = nn.Linear(input_dim, output_dim)
        # 引入旋转角度参数，用于模拟量子旋转门
        self.rotation_angle = nn.Parameter(torch.zeros(1))  # 初始化为0度旋转

    def apply_quantum_rotation(self, x):
        """
        模拟量子旋转门操作，这里简化为一个基于角度的旋转（例如Ry旋转门）。
        实际上，真正的量子旋转门操作涉及复数运算且更复杂，
        这里仅做概念上的模拟。
        """
        # 将旋转角度应用于输入，模拟旋转门效果
        # 注意：这里的实现非常简化，真实量子旋转会更为复杂且通常涉及复数运算
        rotated_x = x * torch.cos(self.rotation_angle) + torch.sin(self.rotation_angle) * torch.sign(x)
        return rotated_x

    def forward(self, x):
        # 先进行线性变换
        linear_output = self.linear_transform(x)

        # 应用量子启发的旋转操作
        rotated_output = self.apply_quantum_rotation(linear_output)

        # 使用sigmoid函数保持输出在(0,1)范围内，模拟概率或门控信号
        return torch.sigmoid(rotated_output)


class QuantumInspiredForgetGate(nn.Module):
    def __init__(self, input_dim):
        super(QuantumInspiredForgetGate, self).__init__()
        # 由于遗忘门主要决定遗忘的程度，我们这里简化为一个线性层直接映射到0-1范围
        # 通过tanh和scale调整来近似这一需求，实际遗忘门设计可能直接使用sigmoid更常见
        self.linear_transform = nn.Linear(input_dim, input_dim)
        # 初始化权重使得初始遗忘倾向不偏向遗忘或保留
        nn.init.uniform_(self.linear_transform.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear_transform.bias)
        # 引入可学习的旋转角度参数，尽管这在传统遗忘门中并不常见
        self.rotation_factor = nn.Parameter(torch.zeros(1))

    def apply_quantum_effect(self, x):
        """
        简化模拟量子效应，通过加权调整x的值。
        实际上，与量子旋转门的直接联系在此简化模型中并不明显。
        """
        # 直接使用线性变换后通过tanh调整范围，并结合潜在的量子效应调整
        # 这里使用tanh是为了确保输出在(-1, 1)之间，然后通过缩放映射到遗忘概率空间
        return torch.tanh(x) * 0.5 + 0.5  # 简化映射到0到1之间

    def forward(self, x):
        # 线性变换后应用“量子”效果，实际上起到调整遗忘程度的作用
        transformed_x = self.linear_transform(x)
        forget_signal = self.apply_quantum_effect(transformed_x)
        # 可选：进一步调整以确保输出严格在0到1之间，尽管tanh+scale已经接近
        # forget_signal = torch.clamp(forget_signal, 0, 1)
        return forget_signal

class QuantumLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(QuantumLayer, self).__init__()

        # 保持线性变换部分
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        self.quantum_gate = QuantumGate(output_size)  # 保持这个模块，虽然它基于权重，但作为示例保留
        self.symbolic_cnot = SymbolicQuantumCNOT()  # 添加象征性的CNOT模块

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 应用线性变换和ReLU
        x = self.linear1(x)
        x = self.relu(x)

        control, target = x.chunk(2, dim=1)  # 假设x可以分为两部分，分别视为"控制"和"目标"
        control, target = self.symbolic_cnot(control, target)
        x = torch.cat((control, target), dim=1)  # 合并回一个张量


        # 继续后续操作
        x = self.quantum_gate(x)  # 示例中保持这个操作
        x = self.linear2(x)
        x = self.tanh(x)

        return x



class QuantumLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, quantum_output_size):
        super(QuantumLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.quantum_layer = QuantumLayer(hidden_dim, quantum_output_size)
        self.quantum_output_gate = QuantumInspiredOutputGate(quantum_output_size, num_class)
        self.quantum_forget_mechanism = QuantumInspiredForgetGate(hidden_dim)
        self.quantum_activation = QuantumInspiredActivation(rotation_axis='x')  # 新增量子启发激活层

    def forward(self, inputs, lengths):
        embeds = self.embedding(inputs)
        lstm_output, _ = self.lstm(embeds)

        # 应用量子启发遗忘机制到LSTM的最终时间步输出
        lstm_output_last_time_step = lstm_output[torch.arange(lstm_output.size(0)), lengths - 1, :]
        forget_factors = self.quantum_forget_mechanism(lstm_output_last_time_step)
        lstm_output_filtered = lstm_output_last_time_step * forget_factors

        # 通过量子层处理
        quantum_output = self.quantum_layer(lstm_output_filtered)

        # 应用量子启发激活层
        activated_output = self.quantum_activation(quantum_output)

        # 通过量子启发输出门
        output_from_quantum_gate = self.quantum_output_gate(activated_output)

        return F.log_softmax(output_from_quantum_gate, dim=-1)
# Set hyperparameters
embedding_dim = 128
hidden_dim = 24
batch_size = 1024
num_epoch = 10
num_class = 2
quantum_input_size = hidden_dim  # Assume quantum input size is the same as LSTM hidden size
quantum_output_size = 32  # Assume quantum output size is 32
learning_rate = 0.001

# Load and preprocess data
train_data, test_data, vocab = build_data("smallsampled_file.csv")
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载模型架构
embedding_dim = 128
hidden_dim = 24
batch_size = 1024
num_epoch = 10
num_class = 2
quantum_input_size = hidden_dim  # Assume quantum input size is the same as LSTM hidden size
quantum_output_size = 32  # Assume quantum output size is 32
learning_rate = 0.001

model = QuantumLSTM(len(vocab), embedding_dim, hidden_dim, num_class, quantum_output_size)
model.to(device)  # 确保模型在正确的设备上

# # # 加载保存的模型参数
# model.load_state_dict(torch.load('qlstmmodelb.pth', map_location=device))
#
# # 设置模型为评估模式，这会关闭诸如Dropout等训练时才需要的行为
# model.eval()
# #
# # 初始化正确预测计数器和总样本数
# total_correct = 0
# total_samples = 0
#
# # 在测试循环之前，初始化两个列表来存储原文本和预测结果
# original_sentences = []
# true_labels = []
# predicted_labels = []
#
# # 修改测试循环以收集信息
# with torch.no_grad():
#     for batch in tqdm(test_data_loader, desc="Testing"):
#         inputs, lengths, targets = [x.to(device) for x in batch]
#
#         # 通过模型进行预测
#         outputs = model(inputs, lengths)
#
#         # 获取预测类别
#         _, preds = torch.max(outputs, 1)
#
#         # 收集当前批次的原文本（需要转换索引回token）、真实标签和预测标签
#         for i in range(targets.size(0)):
#             original_sentence = " ".join(
#                 [vocab.idx_to_token[idx.item()] for idx in inputs[i] if idx.item() != 0])  # 假设0是PAD索引
#             original_sentences.append(original_sentence)
#             true_labels.append("positive" if targets[i].item() == 1 else "negative")
#             predicted_labels.append("positive" if preds[i].item() == 1 else "negative")
#
#         # 继续计算准确率等原有逻辑...
#         correct = (preds == targets).sum().item()
#         total_correct += correct
#         total_samples += targets.size(0)
#
# # 计算并打印测试准确率
# test_accuracy = total_correct / total_samples
# print(f"Test Accuracy using saved model: {test_accuracy:.4f}")
#
# # 打印原文本、真实标签与预测标签
# for sentence, true_label, pred_label in zip(original_sentences, true_labels, predicted_labels):
#     print(f"Original Text: {sentence}\nTrue Label: {true_label}\nPredicted Label: {pred_label}\n{'-' * 50}")
# # 假设 original_sentences, true_labels, predicted_labels 分别存储了原文本、真实标签和预测标签
# # 我们只取前十个样本进行统计
# import matplotlib.pyplot as plt
# # 取前十个样本的标签
# true_labels_subset = true_labels[:60]
# predicted_labels_subset = predicted_labels[:60]
#
# plt.figure(figsize=(10, 5))
# plt.plot(true_labels_subset, label='True Labels', marker='o', linestyle='-')
# plt.plot(predicted_labels_subset, label='Predicted Labels', marker='x', linestyle='--')
#
# plt.title('Comparison of True and Predicted Labels Over Test Samples')
# plt.xlabel('Sample Index')
# plt.ylabel('Label (0 or 1)')
# plt.legend()
# plt.grid(True)
# plt.show()
# 计算测试数据集的准确率
test_losses = []
test_accuracies = []
test_recalls = []
test_f1_scores = []
total_correct = 0
total_samples = 0

for batch in tqdm(test_data_loader, desc="Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    outputs = model(inputs, lengths)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total_correct += correct
    total_samples += targets.size(0)

test_accuracy = total_correct / total_samples
print(f"Test Accuracy: {test_accuracy:.4f}")

# 单独调用模型的代码在自己模拟量子门的文件下面包含了，复制过来修改一下网络名称为当前QLSTM和已保存的qlstm的权重即可运行

# 使用测试数据加载器进行验证
with torch.no_grad():  # 不需要计算梯度，节省内存并加速验证过程
    for batch in tqdm(test_data_loader, desc="Testing"):
        # 准备数据
        inputs, lengths, targets = [x.to(device) for x in batch]

        # 通过模型进行预测
        outputs = model(inputs, lengths)

        # 获取预测类别
        _, predicted = torch.max(outputs, 1)

        # 计算并累加正确预测的数量
        correct = (predicted == targets).sum().item()
        total_correct += correct

        # 累加样本总数
        total_samples += targets.size(0)


# 计算并打印测试准确率
test_accuracy = total_correct / total_samples
test_accuracies.append(test_accuracy)
print(f"Test Accuracy using saved model: {test_accuracy:.4f}")
# Initialize model, optimizer, and loss function
model = QuantumLSTM(len(vocab), embedding_dim, hidden_dim, num_class, quantum_output_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# # 输出到CSV文件
# with open('test_final3.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Epoch', 'Average Loss'])
#     for epoch in range(num_epoch):
#         writer.writerow(
#             [epoch + 1, test_losses[epoch])
# 记录开始训练的时间点
start_time = time.time()

import matplotlib.pyplot as plt

import time
import csv
from sklearn.metrics import precision_recall_fscore_support



# 定义用于记录损失的列表
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

        quantum_gates = [
            model.quantum_layer._modules[name]
            for name, module in model.quantum_layer.named_children()
            if isinstance(module, (QuantumInspiredOutputGate,QuantumGate,QuantumInspiredForgetGate,QuantumInspiredActivation,SymbolicQuantumCNOT))
        ]
        for gate in quantum_gates:#更新量子门的权重，手写模拟更新
            if hasattr(gate, 'weight'):
                gate.weight.grad = gate.weight.grad.clamp(-1, 1)  # 梯度裁剪示例
                gate.weight.data -= learning_rate * gate.weight.grad  # 手动更新权重

        # 更新其他模型参数
        optimizer.step()

        total_loss += loss.item()

    # 保存每个 epoch 的平均损失
    avg_loss = total_loss / len(train_data_loader)
    train_losses.append(avg_loss)

    # 计算训练准确率
    train_accuracy = total_correct / total_samples
    train_accuracies.append(train_accuracy)  # 添加这一行来记录每个epoch的准确率

    # 计算训练准确率
    train_accuracy = total_correct / total_samples
    # 计算召回率和F1分数，需要转换为numpy数组
    predictions = predicted_classes.cpu().numpy()
    targets = targets.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')

    train_recalls.append(recall)
    train_f1_scores.append(f1)

    print(
        f"Epoch {epoch + 1}/{num_epoch}, Average Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f}")

    print(
        f"Epoch {epoch + 1}/{num_epoch}, Average Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f}")


# 输出到CSV文件
with open('train_final3.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Average Loss', 'Train Accuracy', 'Train Recall', 'Train F1'])
    for epoch in range(num_epoch):
        writer.writerow(
            [epoch + 1, train_losses[epoch], train_accuracies[epoch], train_recalls[epoch], train_f1_scores[epoch]])

with open('test_final3.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Test Accuracy'])
    for epoch in range(num_epoch):
        writer.writerow(
            [epoch + 1, test_accuracies[epoch]])

# 记录结束训练的时间点
end_time = time.time()
# 计算训练时长
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))
torch.save(model.state_dict(), 'qlstmmodelb.pth')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()
