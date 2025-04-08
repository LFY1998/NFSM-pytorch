import torch
import torch.nn as nn
from torch.jit import fork, wait


class ChebyTransform(nn.Module):
    def __init__(self, kind='first', degree=1):
        assert kind in ['first', 'second'], "Kind must be either 'first' or 'second'."
        assert degree >= 0, "Degree must be a non-negative integer."
        self.kind = kind
        self.degree = degree
        super().__init__()

    def forward(self, x):
        return self.chebyshev_polynomials(x).view(x.shape[0], -1)

    def chebyshev_polynomials(self, x: torch.Tensor):
        """
        计算输入 x 的 Chebyshev 多项式值。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)
            kind (int): 切比雪夫多项式类型，1 表示第一类，2 表示第二类

        返回:
            torch.Tensor: Chebyshev 多项式值，形状为 (batch_size, in_features, degree + 1)
        """
        # 将 x 缩放到 [-1, 1] 区间
        x = torch.sigmoid(x)

        if self.kind == "first":
            # 计算 arccos(x)
            theta = torch.acos(x)  # 形状为 (batch_size, in_features)

            # 计算每个阶数的 Chebyshev 第一类多项式值
            theta_n = theta.unsqueeze(-1) * torch.arange(self.degree + 1, dtype=x.dtype, device=x.device)

            # 计算 cos(n * arccos(x))，得到 Chebyshev 第一类多项式的值
            T_n = torch.cos(theta_n)  # 形状为 (batch_size, in_features, degree + 1)

        elif self.kind == "second":
            # 计算 arccos(x)
            theta = torch.acos(x)  # 形状为 (batch_size, in_features)

            # 计算每个阶数的 Chebyshev 第二类多项式值
            # 需要 (n + 1) * theta
            n_values = torch.arange(self.degree + 1, dtype=x.dtype, device=x.device) + 1
            theta_n = n_values.unsqueeze(0).unsqueeze(1) * theta.unsqueeze(-1)  # 形状为 (1, batch_size, degree + 1)

            # 计算 sin((n + 1) * arccos(x)) / sin(arccos(x))
            U_n = torch.sin(theta_n) / torch.sin(theta.unsqueeze(-1))  # 形状为 (batch_size, degree + 1)

            # 处理 sin(theta) 为 0 的情况，避免除零
            U_n[torch.isnan(U_n)] = 0

        else:
            raise ValueError("kind must be either 1st or 2nd for Chebyshev polynomials.")

        return T_n if self.kind == "first" else U_n


class ChebyMLP(nn.Module):
    def __init__(self, hidden_layers, dropout_prob=0.0, degree=5, chebyorder="second"):
        """
        初始化普通 MLP 模型并自定义权重初始化，并在隐藏层中加入 Dropout。
        :param hidden_layers: 隐藏层维度列表，例如 [3, h1, h2, ..., 1]，其中输入维度为 3
        :param dropout_prob: Dropout 概率，默认值为 0.5
        """
        super(ChebyMLP, self).__init__()

        layers = []

        # 构建隐藏层，不在输入层使用 Dropout
        for i in range(len(hidden_layers) - 1):
            layers.append(ChebyTransform(kind=chebyorder, degree=degree))
            layers.append(nn.BatchNorm1d((degree + 1) * hidden_layers[i]))
            layers.append(nn.Linear((degree + 1) * hidden_layers[i], hidden_layers[i + 1]))
            # layers.append(nn.LayerNorm(hidden_layers[i+1]))
            if i > 0:  # 在隐藏层中添加 Dropout，输入层不加
                layers.append(nn.Dropout(dropout_prob))


        # 将所有层组合为序列模型
        self.mlp = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def forward(self, P):
        # P 的形状为 [Nx, Ny, Nz, 3]，需要先将其展平为 [Nx * Ny * Nz, 3]
        # P_flat = P.view(-1, 3)  # 将输入展平为二维

        # 计算输出
        output = self.mlp(P)

        # # 将结果 reshape 回 [Nx, Ny, Nz]
        # output = output.view(*P.shape[:-1], 2)  # 恢复为三维张量形状

        return output

    def _initialize_weights(self):
        """
        初始化网络的权重
        使用 Xavier 或 Kaiming 初始化
        """
        torch.manual_seed(0)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                # 使用 Kaiming 初始化，适合 ReLU 激活函数
                nn.init.orthogonal_(layer.weight)

                # 可以选择性地将 bias 初始化为 0
                # if layer.bias is not None:
                #     nn.init.constant_(layer.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, hidden_layers, dropout_prob=0.0):
        """
        初始化多层感知机
        :param hidden_layers: 包含输入层和输出层的隐藏层节点数量列表 (e.g., [input_dim, hidden1, ..., output_dim])
        :param dropout_prob: dropout 的概率，默认为 0.0
        """
        super(MLP, self).__init__()

        if len(hidden_layers) < 2:
            raise ValueError("hidden_layers must contain at least input and output dimensions.")

        self.hidden_layers = hidden_layers
        self.dropout_prob = dropout_prob

        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))  # 添加线性层
            if i < len(hidden_layers) - 2:  # 输出层之后不需要激活和 Dropout
                layers.append(nn.ReLU())  # 使用 ReLU 激活函数
                if dropout_prob > 0.0:
                    layers.append(nn.Dropout(dropout_prob))  # 添加 Dropout

        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化网络的权重
        使用 Xavier 或 Kaiming 初始化
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):  # 仅对 Linear 层初始化
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # Kaiming 初始化
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # 将偏置初始化为 0

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        return self.model(x)


class NFModel(nn.Module):
    def __init__(self, hidden_layers, M_order, ChebyOrder="first", dec_index=None, hidden_layers_enc=None, mode=None,
                 parallel_method="jit", degree=7):
        """
        parallel_method: str, "jit" 或 "stream"，用于选择并行方法
        """
        super(NFModel, self).__init__()
        self.mode = mode
        self.parallel_method = parallel_method  # 设置并行方式
        self.share_encoder = nn.Identity()
        if hidden_layers_enc is not None:
            if "Cheby" in mode:
                self.share_encoder = ChebyMLP(hidden_layers_enc,
                                              chebyorder=ChebyOrder, degree=degree)
            elif "noCheby" in mode:
                self.share_encoder = MLP(hidden_layers_enc)
        self.model = nn.ModuleDict()
        for ck in dec_index:
            if mode == "Cheby-M-order":
                self.model[ck] = ChebyMLP([*hidden_layers[:-1], 3 * M_order * hidden_layers[-1]],
                                          chebyorder=ChebyOrder, degree=degree)
            elif mode == "Cheby-standard":
                self.model[ck] = ChebyMLP(hidden_layers,
                                          chebyorder=ChebyOrder, degree=degree)
            elif mode == "noCheby-standard":
                self.model[ck] = MLP(hidden_layers)

            elif mode == "noCheby-M-order":
                self.model[ck] = MLP([*hidden_layers[:-1], 3 * M_order * hidden_layers[-1]])
        self.M_order = M_order
        self.hidden = hidden_layers

    def forward(self, P, dec_idx, train_encoder=True):
        # 将 P 展平为 [Nx * Ny * Nz, 3]
        P_flat = P.view(-1, 3)
        # 共享编码
        if train_encoder:
            P_flat_enc = self.share_encoder(P_flat)
        else:
            P_flat_enc = self.share_encoder(P_flat).detach()
        if self.parallel_method == "jit":
            return self._forward_jit(P, P_flat_enc, dec_idx)
        elif self.parallel_method == "stream":
            return self._forward_stream(P, P_flat_enc, dec_idx)
        else:
            raise ValueError("Invalid parallel_method. Choose either 'jit' or 'stream'.")

    def _forward_jit(self, P, P_flat_enc, dec_idx):
        y_out = torch.zeros([len(dec_idx), P_flat_enc.shape[0], self.hidden[-1]], device=P_flat_enc.device)
        futures = []
        for i in dec_idx:
            model_i = self.model[str(i)]
            if self.mode == "Cheby-M-order" or self.mode == "noCheby-M-order":
                futures.append(fork(self._compute_M_order, model_i, P_flat_enc))
            elif self.mode == "Cheby-standard" or self.mode == "noCheby-standard":
                futures.append(fork(model_i, P_flat_enc))
        y1_minus_avg_list = []
        y1_list = []
        for step, future in enumerate(futures):
            y1 = wait(future)
            if self.mode == "Cheby-M-order" or self.mode == "noCheby-M-order":
                y1_avg = self._average_grouped_coordinates(P, y1)
                y1_minus_avg_list.append(torch.norm(y1 - y1_avg)**2/P.shape[0])
                y1_list.append(torch.norm(y1)**2/P.shape[0])
                y2 = y1_avg[:, 0, :, :] * y1_avg[:, 1, :, :] * y1_avg[:, 2, :, :]
                y_out[step, :, :] = torch.sum(y2, dim=1)
            elif self.mode == "Cheby-standard" or self.mode == "noCheby-standard":
                y_out[step, :, :] = y1

        if self.mode == "Cheby-M-order" or self.mode == "noCheby-M-order":
            return y_out, (torch.stack(y1_minus_avg_list), y1_list)
        elif self.mode == "Cheby-standard" or self.mode == "noCheby-standard":
            return y_out, 1

    def _forward_stream(self, P, P_flat_enc, dec_idx):
        streams = [torch.cuda.Stream() for _ in dec_idx]
        results = [None] * len(dec_idx)
        y1_minus_avg_list = []
        for step, (i, stream) in enumerate(zip(dec_idx, streams)):
            with torch.cuda.stream(stream):
                model_i = self.model[str(i)]
                y1 = model_i(P_flat_enc)
                if self.mode == "Cheby-M-order" or self.mode == "noCheby-M-order":
                    y1 = y1.view(-1, 3, self.M_order, self.hidden[-1])
                    y1_avg = self._average_grouped_coordinates(P, y1)
                    y1_minus_avg_list.append(y1 - y1_avg)
                    y2 = y1_avg[:, 0, :, :] * y1_avg[:, 1, :, :] * y1_avg[:, 2, :, :]
                    results[step] = torch.sum(y2, dim=1)
                elif self.mode == "Cheby-standard" or self.mode == "noCheby-standard":
                    results[step] = y1

        torch.cuda.synchronize()
        y_out = torch.stack(results)

        if self.mode == "Cheby-M-order" or self.mode == "noCheby-M-order":
            return y_out, torch.stack(y1_minus_avg_list, dim=1)
        elif self.mode == "Cheby-standard" or self.mode == "noCheby-standard":
            return y_out, 1

    def _compute_M_order(self, model_i, P_flat_enc):
        y1 = model_i(P_flat_enc)
        return y1.view(-1, 3, self.M_order, self.hidden[-1])

    def _average_grouped_coordinates(self, P_flat, y1):
        """
        对于每个维度（x, y, z），将 y1 的值按相同的坐标分组，并计算均值。
        :param P_flat: 展平后的输入坐标，形状为 [N, 3]
        :param y1: MLP 的输出，形状为 [N, 3, R, hidden_layers[-1]]
        :return: 更新后的 y1
        """
        # 初始化一个与 y1 相同形状的张量来存储结果
        y1_avg = torch.zeros_like(y1)
        # y1_group = []
        # 遍历每个坐标维度 (x, y, z)
        for dim in range(3):
            # 获取该维度的坐标索引
            coords = P_flat[:, dim]

            # 找出唯一值及其反向映射
            unique_vals, inverse_indices = torch.unique(coords, return_inverse=True)

            # 初始化用于累加的张量和计数
            num_unique = unique_vals.size(0)
            summed_y1 = torch.zeros(num_unique, self.M_order, self.hidden[-1], device=y1.device)
            counts = torch.zeros(num_unique, 1, 1, device=y1.device)
            inverse_indices = inverse_indices.to(y1.device)
            # 使用 index_add_ 进行分组累加
            summed_y1.index_add_(0, inverse_indices, y1[:, dim, :, :])

            # 计算每组的计数
            counts.index_add_(0, inverse_indices,
                              torch.ones_like(coords, dtype=torch.float32).unsqueeze(1).unsqueeze(2).to(y1.device))

            # 计算每组的平均值
            mean_y1 = summed_y1 / counts

            # 将平均值扩展回原始形状
            y1_avg[:, dim, :, :] = mean_y1[inverse_indices, :, :]
            # y1_group.append(mean_y1[None])

        return y1_avg


# class NFModelSep(nn.Module):


def custom_loss(y1, batch_indices):
    """
    计算约束损失，使得相同 x、y、z 坐标的 y1[:, 0, :, :]，y1[:, 1, :, :]，y1[:, 2, :, :] 应该趋于相同。
    :param y1: 模型输出，形状为 [batch_size, 3, R, hidden_layers[-1]]
    :param batch_indices: 坐标索引，形状为 [batch_size, 3]，记录了 (x, y, z) 坐标的索引
    :return: 约束损失值
    """
    loss = 0.0

    # 遍历 x, y, z 维度 (dim 取值为 0, 1, 2 分别对应 x, y, z)
    for dim in range(3):
        # 获取当前维度上的坐标索引
        indices = batch_indices[:, dim]

        # 获取相同坐标的唯一值及其反向映射 (即每个坐标点在 unique 值中的索引)
        unique_vals, inverse_indices = torch.unique(indices, return_inverse=True)
        inverse_indices = inverse_indices.to(y1.device)

        # 对于每个维度 dim，按相同坐标点分组计算均值
        y1_dim = y1[:, :, dim, :, :]  # 取出对应的 dim 维度 (x, y 或 z) 的 y1

        # unique_vals 的数量决定了有多少个分组
        num_unique = unique_vals.size(0)

        # 初始化相同坐标的均值张量，形状为 [BS, num_unique, R, hidden_layers[-1]]
        summed_y1 = torch.zeros(y1.shape[0], num_unique, y1_dim.size(2), y1_dim.size(3)).to(y1.device)

        # counts 只需要跟 unique_vals 的数量相关，用来记录每个组的累加次数
        counts = torch.zeros(num_unique, dtype=torch.float32).to(y1.device)

        # 按照相同的坐标进行累加，计算分组和
        summed_y1.index_add_(1, inverse_indices, y1_dim)

        # 计算分组中的元素个数
        counts.index_add_(0, inverse_indices, torch.ones_like(indices, dtype=torch.float32).to(y1.device))

        # 计算分组均值
        mean_y1 = summed_y1 / counts[None, :, None, None]

        # 将分组均值广播回原始形状
        expanded_mean_y1 = mean_y1[:, inverse_indices, :, :]

        # 计算每个分组内的 MSE 损失
        loss += nn.MSELoss(reduction='none')(y1_dim, expanded_mean_y1).mean(dim=(1, 2, 3)).sum(dim=0)

    return loss


class SimplyNFModel(torch.nn.Module):
    def __init__(self, hidden_layers, M_order, ChebyOrder="first", dec_index=None, hidden_layers_enc=None, mode=None,
                 parallel_method="jit", degree=7):
        """
        parallel_method: str, "jit" 或 "stream"，用于选择并行方法
        """
        super(SimplyNFModel, self).__init__()
        self.mode = mode
        self.parallel_method = parallel_method  # 设置并行方式
        self.share_encoder = ChebyMLP(hidden_layers_enc, chebyorder=ChebyOrder, degree=degree)

        self.model = nn.ModuleDict()
        for ck in dec_index:
            self.model[ck] = nn.ModuleDict({
                "xdec": ChebyMLP([*hidden_layers[:-1], M_order * hidden_layers[-1]]),
                "ydec": ChebyMLP([*hidden_layers[:-1], M_order * hidden_layers[-1]]),
                "zdec": ChebyMLP([*hidden_layers[:-1], M_order * hidden_layers[-1]])})

        self.M_order = M_order
        self.hidden = hidden_layers
        self.hidden_layers_enc = hidden_layers_enc

    def forward(self, P, Px, Py, Pz, dec_idx, train_encoder=True):
        # 将 P 展平为 [Nx * Ny * Nz, 3]
        P_flat = torch.cat((Px, Py, Pz), dim=0)
        # 共享编码
        if train_encoder:
            P_flat_enc = self.share_encoder(P_flat[:, None])
        else:
            P_flat_enc = self.share_encoder(P_flat).detach()
        P_flat_enc = P_flat_enc.view(-1, 3, self.hidden_layers_enc[-1])
        if self.parallel_method == "jit":
            return self._forward_jit(P, P_flat_enc, dec_idx)
        else:
            raise ValueError("Invalid parallel_method. Choose either 'jit' or 'stream'.")

    def _forward_jit(self, P, P_flat_enc, dec_idx):
        y_out = torch.zeros([len(dec_idx), P.shape[0], self.hidden[-1]], device=P_flat_enc.device)
        futures = []
        for i in dec_idx:
            model_i = self.model[str(i)]
            futures.append(fork(self._compute_M_order, model_i, P_flat_enc))

        y1_minus_avg_list = []
        for step, future in enumerate(futures):
            y1 = wait(future)
            x_dec = y1[:, 0]
            y_dec = y1[:, 1]
            z_dec = y1[:, 2]
            y_out[step] = torch.sum(
                x_dec[P[:, 0], :, :] * y_dec[P[:, 1], :, :] * z_dec[P[:, 2], :, :],
                dim=1  # 在 M 维上求和
            )
        return y_out


    def _compute_M_order(self, model_i, P_flat_enc):
        xdec = model_i["xdec"](P_flat_enc[:, 0, :])[:, None, :]
        ydec = model_i["ydec"](P_flat_enc[:, 1, :])[:, None, :]
        zdec = model_i["zdec"](P_flat_enc[:, 2, :])[:, None, :]
        y1 = torch.cat([xdec, ydec, zdec], dim=1)
        return y1.view(-1, 3, self.M_order, self.hidden[-1])
