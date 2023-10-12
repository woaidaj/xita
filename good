class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        
        self.coordatt3 = CoordAtt(64, 64)  # Adjusted for 64 channels
        
        # Only one max pooling layer after the last convolution
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Another CoordAtt after max pooling
        self.coordatt_after_pool = CoordAtt(64, 64)  # Assuming the channel count remains 64 after max pooling
        
        # Fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(34816, 256)  # Adjust the dimension accordingly

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.coordatt3(x)  # Using CoordAtt
        
       # Max pooling after the last convolution and CoordAtt
        
      # Using the second CoordAtt after max pooling这次是去掉后面池化  强化模型能力
    
        
        x = self.flatten(x)
        x = self.fc(x)
        x = x.unsqueeze(1)
        return x

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv_net = ConvNet()  # Use the shared ConvNet

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
