import numpy as np
import mne
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split,StratifiedKFold
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.init as init
import matplotlib.pyplot as plt
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import mne
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split,StratifiedKFold
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.init as init
import matplotlib.pyplot as plt
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import torch
from torch import nn
import torch
import torch.optim as optim
import random
import os
import pickle
import os
import re
import os
import re
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x






class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.res_block1 = ResidualBlock(1, 16, kernel_size=5, stride=2, padding=2)
        self.res_block2 = ResidualBlock(16, 32, kernel_size=5, stride=2, padding=2)

        # Add an additional CBAM layer
        self.cbam1 = CBAM(in_channel=32)
        
        self.res_block3 = ResidualBlock(32, 64, kernel_size=5, stride=2, padding=2)
        
        # Original CBAM layer
        self.cbam2 = CBAM(in_channel=64)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*17*32, 256)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Use the additional CBAM layer
        x = self.cbam1(x)
        
        x = self.res_block3(x)

        # Original CBAM
        x = self.cbam2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        x = x.unsqueeze(1)
        return x


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv_net = ConvNet()  # 使用一个共享的ConvNet

    def forward(self, x):
        outputs = []
        for i in range(20):
            channel_data = x[:, i:i+1, :, :]
            output = self.conv_net(channel_data)
            outputs.append(output.squeeze(1))

        return torch.stack(outputs, dim=1)








class BiLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.initialize_weights()  # 初始化权重

        # 定义前向 LSTM 的权重
        self.input_weights_i = nn.Linear(input_size, hidden_size)
        self.input_weights_f = nn.Linear(input_size, hidden_size)
        self.input_weights_o = nn.Linear(input_size, hidden_size)
        self.input_weights_s = nn.Linear(input_size, hidden_size)

        self.hidden_weights_i = nn.Linear(hidden_size, hidden_size)
        self.hidden_weights_f = nn.Linear(hidden_size, hidden_size)
        self.hidden_weights_o = nn.Linear(hidden_size, hidden_size)
        self.hidden_weights_s = nn.Linear(hidden_size, hidden_size)

        self.neighbor_weights_i = nn.Linear(4, hidden_size)
        self.neighbor_weights_f = nn.Linear(4, hidden_size)
        self.neighbor_weights_o = nn.Linear(4, hidden_size)
        self.neighbor_weights_s = nn.Linear(4, hidden_size)

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


    # ... 其余代码保持不变

    def forward(self, x, hidden, cell, neighbors):
        input_gate = torch.sigmoid(self.input_weights_i(x) + self.hidden_weights_i(hidden) + self.neighbor_weights_i(neighbors))
        forget_gate = torch.sigmoid(self.input_weights_f(x) + self.hidden_weights_f(hidden) + self.neighbor_weights_f(neighbors))
        output_gate = torch.sigmoid(self.input_weights_o(x) + self.hidden_weights_o(hidden) + self.neighbor_weights_o(neighbors))

        cell_candidate = torch.tanh(self.input_weights_s(x) + self.hidden_weights_s(hidden) + self.neighbor_weights_s(neighbors))
        next_cell = forget_gate * cell + input_gate * cell_candidate
        next_hidden = output_gate * torch.tanh(next_cell)

        return next_hidden, next_cell


class DCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rows, cols, num_neurons, device, my_network):
        super(DCRNN, self).__init__()
        # Code omitted for brevity
        self.my_network = my_network
        self.rows = rows
        self.cols = cols
        self.hidden_size = hidden_size
        self.device = device  # 添加设备属性

        self.cell_fwd = BiLSTMCell(input_size, hidden_size)
        self.cell_bwd = BiLSTMCell(input_size, hidden_size)

        self.FF = nn.Linear(200, num_neurons)
        self.output = nn.Linear(num_neurons, 2)
        self.activation = nn.Softmax(dim=1)
        self.initialize_weights()  # 初始化权重

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'FF' in name or 'output' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
    def forward(self, x):
        x = self.my_network(x)
        x = x.to(self.device)
        x = torch.transpose(x, 0, 1)
        ch, bs, seq_len = x.shape
        hidden, cell = self.init_hidden_cell(bs)

        H_fwd, H_bwd = [], []

        # 转换 hidden 和 cell 为 list
        hidden = [list(h) for h in hidden]
        cell = [list(c) for c in cell]

        # 正向传播
        for t in range(seq_len):
            last_hidden_states = []  # 用于存储最后一个时间步长的所有单元的 next_hidden 值

            for i in range(self.rows * self.cols):
                channel_idx = i
                x_t = x[channel_idx, :, t].unsqueeze(-1)
                neighbors = self.get_neighbor_hidden(hidden, i, bs, direction='forward')
                next_hidden, next_cell = self.cell_fwd(x_t, hidden[i][0], cell[i][0], neighbors)

                hidden_forward, hidden_backward = hidden[i]
                hidden_forward = next_hidden
                hidden[i] = (hidden_forward, hidden_backward)

                cell_forward, cell_backward = cell[i]
                cell_forward = next_cell
                cell[i] = (cell_forward, cell_backward)

                if t == seq_len - 1:
                    last_hidden_states.append(next_hidden)  # 当时间步长为最后一个时，将当前单元的 next_hidden 添加到 last_hidden_states 列表中

            if t == seq_len - 1:
                H_fwd = torch.stack(last_hidden_states, dim=1)  # 拼接 last_hidden_states 中的张量并创建 H_fwd

        # 反向传播
        for t in reversed(range(seq_len)):
            last_hidden_states = []  # 用于存储最后一个时间步长的所有单元的 next_hidden 值

            for i in range(self.rows * self.cols):
                channel_idx = i
                x_t = x[channel_idx, :, t].unsqueeze(-1)
                neighbors = self.get_neighbor_hidden(hidden, i, bs, direction='backward')
                next_hidden, next_cell = self.cell_bwd(x_t, hidden[i][1], cell[i][1], neighbors)

                hidden_forward, hidden_backward = hidden[i]
                hidden_backward = next_hidden
                hidden[i] = (hidden_forward, hidden_backward)

                cell_forward, cell_backward = cell[i]
                cell_backward = next_cell
                cell[i] = (cell_forward, cell_backward)

                if t == 0:  # 当时间步长为第一个（反向传播中的最后一个）时
                    last_hidden_states.append(next_hidden)

            if t == 0:
                H_bwd = torch.stack(last_hidden_states, dim=1)  # 拼接 last_hidden_states 中的张量并创建 H_bwd

        # 转换 hidden 和 cell 回 tuple
        hidden = [tuple(h) for h in hidden]
        cell = [tuple(c) for c in cell]

        # 将正向和反向隐藏状态合并
        H = torch.cat((H_fwd, H_bwd), dim=2)
        H = H.view(bs, -1)  # 将 H 的形状从 (32, 20, 10) 改为 (32, 200)

        # 接下来是前馈子网络和输出层的处理
        FF_out = torch.sigmoid(self.FF(H))
        output_out = self.activation(self.output(FF_out))
        return output_out

    def init_hidden_cell(self, batch_size):
        hidden_cell = [(
            (torch.randn(batch_size, self.hidden_size).to(self.device) * 0.01,
             torch.randn(batch_size, self.hidden_size).to(self.device) * 0.01),
            (torch.randn(batch_size, self.hidden_size).to(self.device) * 0.01,
             torch.randn(batch_size, self.hidden_size).to(self.device) * 0.01)
        ) for _ in range(self.rows * self.cols)]
        return list(zip(*[zip(*hc) for hc in hidden_cell]))

    def get_neighbor_hidden(self, hidden, index, batch_size, direction):
        row = index // self.cols
        col = index % self.cols

        if direction == 'forward':
            neighbor_indices = [
                (row - 1) * self.cols + col,
                (row + 1) * self.cols + col,
                row * self.cols + col - 1,
                row * self.cols + col + 1
            ]
        elif direction == 'backward':
            neighbor_indices = [
                (row - 1) * self.cols + col,
                (row + 1) * self.cols + col,
                row * self.cols + col + 1,
                row * self.cols + col - 1
            ]
        else:
            raise ValueError("Invalid direction")

        neighbors = []
        G = self.hidden_size 

        for idx in neighbor_indices:
            if 0 <= idx < self.rows * self.cols:
                if direction == 'forward':
                        hidden_forward, hidden_backward = hidden[idx]  # 提取 hidden_forward 和 hidden_backward
                        # 提取每一行的第四列
                        neighbor_output = hidden_forward[:, 3].unsqueeze(1).unsqueeze(2)  # 这将创建一个形状为[32, 1, 1]的张量
                        # 接下来，您可以将 G_values 添加到您的数据结构中或进行其他操作。

                else:
                    hidden_forward, hidden_backward = hidden[idx]  # 提取 hidden_forward 和 hidden_backward
                    # 提取每一行的第四列
                    neighbor_output = hidden_backward[:, 3].unsqueeze(1).unsqueeze(2)  # 这将创建一个形状为[32, 1, 1]的张量
                    # 接下来，您可以将 G_values 添加到您的数据结构中或进行其他操作。
                neighbors.append(neighbor_output)
            else:
                neighbors.append(torch.zeros(batch_size, 1, 1).to(self.device))

        return torch.cat(neighbors, dim=-1).squeeze(1)






class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.mse_loss(y_pred, y_true) / 2


class MyDataset(Dataset):
    def __init__(self, file_paths, file_list):
        self.file_paths = file_paths
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 选择对应的文件路径
        file_path = self.file_paths[idx % len(self.file_paths)]
        X, y = self.load_data(os.path.join(file_path, self.file_list[idx]))
        X = X.view(-1, *X.shape[2:])
        y = y.view(-1, *y.shape[2:])
        return X, y

    @staticmethod
    def load_data(file_name):
        try:
            with open(file_name, 'rb') as f:
                X = pickle.load(f)
            y_file_name = file_name.replace('X', 'y')
            with open(y_file_name, 'rb') as f:
                y = pickle.load(f)
        except Exception as e:
            print(f"加载文件 {file_name} 时出错：{e}")
            X, y = None, None
        return torch.tensor(X), torch.tensor(y)
    
    


import os
import re
from torch.utils.data import DataLoader

# 假设你已经在某处定义了 MyDataset 类

import re
import os

def main():
    data_dirs = [
    
        

        
       
         '/root/autodl-tmp/batches5/',
        '/root/autodl-tmp/batches4/',
        '/root/autodl-tmp/batches3/',
        '/root/autodl-tmp/batches2/',
        '/root/autodl-tmp/batches1/'
       
    ]

    for i in range(len(data_dirs)):
        # 使用除当前索引 i 之外的所有数据目录作为训练和验证数据
        # 执行留一法交叉验证
        train_validation_dirs = data_dirs[:i] + data_dirs[i + 1:]
        # 使用索引 i 的数据目录作为测试数据
        test_dir = data_dirs[i]

        # 在每轮循环开始时，重置列表以收集每个文件夹的数据文件和标签文件的路径
        all_train_data_files = []
        all_train_label_files = []
        all_valid_data_files = []
        all_valid_label_files = []

        # 遍历训练和验证数据目录
        for data_dir in train_validation_dirs:
            # 列出目录中的所有文件
            files = os.listdir(data_dir)
            # 按数字对文件进行排序（这里假设文件名包含数字）
            files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

            # 在开始新的迭代时清空 data_files 和 label_files
            data_files = []
            label_files = []

            # 遍历目录中的文件
            for file_name in files:
                # 如果文件名以 “X_batch_” 开头，将其添加到数据文件列表中
                if file_name.startswith('X_batch_'):
                    data_files.append(os.path.join(data_dir, file_name))
                # 如果文件名以 “y_batch_” 开头，将其添加到标签文件列表中
                elif file_name.startswith('y_batch_'):
                    label_files.append(os.path.join(data_dir, file_name))

            # 区分 seizure 和 non-seizure 文件
            seizure_files = data_files[:1798]
            non_seizure_files = data_files[1798:]
            seizure_labels = label_files[:1798]
            non_seizure_labels = label_files[1798:]

            # 计算 seizure 类别的训练和验证分割点
            train_split_seizure = int(0.8 * len(seizure_files))
            # 将 seizure 文件分割为训练和验证集
            train_files_seizure = seizure_files[:train_split_seizure]
            valid_files_seizure = seizure_files[train_split_seizure:]
            train_labels_seizure = seizure_labels[:train_split_seizure]
            valid_labels_seizure = seizure_labels[train_split_seizure:]

            # 计算 non-seizure 类别的训练和验证分割点
            train_split_non_seizure = int(0.8 * len(non_seizure_files))
            # 将 non-seizure 文件分割为训练和验证集
            train_files_non_seizure = non_seizure_files[:train_split_non_seizure]
            valid_files_non_seizure = non_seizure_files[train_split_non_seizure:]
            train_labels_non_seizure = non_seizure_labels[:train_split_non_seizure]
            valid_labels_non_seizure = non_seizure_labels[train_split_non_seizure:]

            # 将当前文件夹的文件路径添加到总的训练和验证文件列表中
            all_train_data_files += train_files_seizure + train_files_non_seizure
            all_train_label_files += train_labels_seizure + train_labels_non_seizure
            all_valid_data_files += valid_files_seizure + valid_files_non_seizure
            all_valid_label_files += valid_labels_seizure + valid_labels_non_seizure

        # 在这里，你可以使用 all_train_data_files、all_train_label_files、all_valid_data_files、all_valid_label_files 和 test_dir 进行训练、验证和测试



        train_dataset = MyDataset(file_paths=data_dirs, file_list=all_train_data_files)
        validation_dataset = MyDataset(file_paths=data_dirs, file_list=all_valid_data_files)
        batch_size =160
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # validation_loader = ...
    # test_loader = ...
        # 获取测试集文件
        test_data_files = []
        test_label_files = []
        files = os.listdir(test_dir)
        files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

        # 分离数据文件和标签文件
        for file_name in files:
            if file_name.startswith('X_batch_'):
                test_data_files.append(os.path.join(test_dir, file_name))
            elif file_name.startswith('y_batch_'):
                test_label_files.append(os.path.join(test_dir, file_name))

        # 创建 DataLoader 用 all_train_data_files 和 all_train_label_files 作为训练集
        # 用 all_valid_data_files 和 all_valid_label_files 作为验证集
        # 用 test_data_files 和 test_label_files 作为测试集


        test_dataset = MyDataset(file_paths=[test_dir], file_list=test_data_files)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型、损失函数和优化器
        input_size = 1
        hidden_size = 5
        rows = 4
        cols = 5
        num_neurons = 50
        epochs = 11
        learning_rate = 0.001

        my_network = MyNetwork()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DCRNN(input_size, hidden_size, rows, cols, num_neurons, device, my_network)
        model = model.to(device)
        loss_fn = CustomMSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_valid_loss = float('inf')

        # 初始化列表以存储指标
        # 初始化列表以存储指标
        def sensitivity_specificity(targets, predictions):
            tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            return sensitivity, specificity
       
        train_accuracies = []
        train_sensitivities = []
        train_specificities = []
        train_aucs = []

        valid_accuracies = []
        valid_sensitivities = []
        valid_specificities = []
        valid_aucs = []
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # 训练和验证
        for epoch in range(epochs):
            # 训练
            model.train()
            total_train_loss = 0
            total_batches = 0
            y_train_true = []
            y_train_pred = []
            y_train_scores = [] 
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_hat = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                # 将预测的概率转换为类别标签
                y_hat_classes = torch.argmax(y_hat, dim=1)
                y_batch_classes = torch.argmax(y_batch, dim=1)

                y_train_true.extend(y_batch_classes.tolist())
                y_train_pred.extend(y_hat_classes.tolist())
                y_train_scores.extend(y_hat[:, 1].cpu().detach().numpy())

                total_batches += 1

            train_loss = total_train_loss / total_batches

            # 计算训练评估指标
            train_accuracy = accuracy_score(y_train_true, y_train_pred)
            train_sensitivity, train_specificity = sensitivity_specificity(y_train_true, y_train_pred)
            train_auc = roc_auc_score(y_train_true, y_train_scores)
            # 记录训练评估指标
            train_accuracies.append(train_accuracy)
            train_sensitivities.append(train_sensitivity)
            train_specificities.append(train_specificity)
            train_aucs.append(train_auc)
            # 验证
            model.eval()
            total_valid_loss = 0
            total_valid_batches = 0
            y_valid_true = []
            y_valid_pred = []
            y_valid_scores = []
            with torch.no_grad():
                for X_batch, y_batch in validation_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_hat = model(X_batch)
                    loss = loss_fn(y_hat, y_batch)
                    total_valid_loss += loss.item()

                    # 将预测的概率转换为类别标签
                    y_hat_classes = torch.argmax(y_hat, dim=1)
                    y_batch_classes = torch.argmax(y_batch, dim=1)

                    y_valid_true.extend(y_batch_classes.tolist())
                    y_valid_pred.extend(y_hat_classes.tolist())
                    y_valid_scores.extend(y_hat[:, 1].cpu().numpy())

                    # 增加批次计数
                    total_valid_batches += 1

            # 计算平均验证损失
            valid_loss = total_valid_loss / total_valid_batches

            # 如果验证损失有所降低，保存模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_model.pt')

         
             # 计算验证评估指标
            valid_accuracy = accuracy_score(y_valid_true, y_valid_pred)
            valid_sensitivity, valid_specificity = sensitivity_specificity(y_valid_true, y_valid_pred)
            valid_auc = roc_auc_score(y_valid_true, y_valid_scores)

  
         
            # 记录验证评估指标
            valid_accuracies.append(valid_accuracy)
            valid_sensitivities.append(valid_sensitivity)
            valid_specificities.append(valid_specificity)
            valid_aucs.append(valid_auc)
            
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")
                
            print(f"Train Accuracy: {train_accuracy}, Sensitivity: {train_sensitivity}, Specificity: {train_specificity}, AUC: {train_auc}")
            print(f"Valid Accuracy: {valid_accuracy}, Sensitivity: {valid_sensitivity}, Specificity: {valid_specificity}, AUC: {valid_auc}")


            # 在测试集上评估最佳模型
        model.load_state_dict(torch.load('best_model.pt'))

        # 测试
        model.eval()
        test_loss = 0
        y_test_true = []
        y_test_pred = []
        y_test_scores = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_hat = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                test_loss += loss.item()

                # 将预测的概率转换为类别标签
                y_hat_classes = torch.argmax(y_hat, dim=1)
                y_batch_classes = torch.argmax(y_batch, dim=1)

                y_test_true.extend(y_batch_classes.tolist())
                y_test_pred.extend(y_hat_classes.tolist())
                y_test_scores.extend(y_hat[:, 1].cpu().numpy())

            test_loss /= len(test_loader)

   
        
        # 计算测试评估指标
        test_accuracy = accuracy_score(y_test_true, y_test_pred)
        test_sensitivity, test_specificity = sensitivity_specificity(y_test_true, y_test_pred)
        test_auc = roc_auc_score(y_test_true, y_test_scores)

      
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}, Sensitivity: {test_sensitivity}, Specificity: {test_specificity}, AUC: {test_auc}")




if __name__ == "__main__":
    main()