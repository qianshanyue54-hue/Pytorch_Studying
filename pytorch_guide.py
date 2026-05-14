"""
PyTorch 常用功能完整指南
========================
涵盖：张量操作、自动求导、神经网络、优化器、数据加载、GPU训练、模型保存/加载等
"""


import sys
print(f"当前运行的 Python 路径: {sys.executable}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms, models
import numpy as np
import math

# =============================================================================
# 一、张量 (Tensor) 基础操作
# =============================================================================

# --- 1.1 创建张量 ---

# 从列表创建
a = torch.tensor([1, 2, 3])                    # 从Python列表创建一维张量
b = torch.tensor([[1, 2], [3, 4]])             # 从嵌套列表创建二维张量

# 特殊张量
z = torch.zeros(3, 4)                          # 全0张量，形状 (3, 4)
o = torch.ones(2, 3)                           # 全1张量，形状 (2, 3)
e = torch.eye(3)                               # 3x3 单位矩阵
r = torch.rand(2, 3)                           # [0, 1) 均匀分布随机张量
rn = torch.randn(2, 3)                         # 标准正态分布 N(0,1) 随机张量
ri = torch.randint(0, 10, (3, 4))              # [0, 10) 随机整数张量，形状 (3,4)
f = torch.full((2, 3), 7.0)                    # 用 7.0 填充的 (2,3) 张量

# 从序列创建
ar = torch.arange(0, 10, 2)                    # 从0到10步长为2 → [0,2,4,6,8]
lin = torch.linspace(0, 1, 5)                  # [0,1] 等间隔取5个点 → [0, 0.25, 0.5, 0.75, 1.0]
log = torch.logspace(0, 2, 3, base=10)         # 10^0 到 10^2 对数等间隔 → [1, 10, 100]

# 与已有张量相似
x = torch.randn(3, 4)
same_shape = torch.zeros_like(x)                # 形状与x相同的全0张量
same_shape2 = torch.ones_like(x)                # 形状与x相同的全1张量
same_shape3 = torch.randn_like(x)               # 形状与x相同的随机张量

# 从 NumPy 互转
np_arr = np.array([1, 2, 3])
from_np = torch.from_numpy(np_arr)             # NumPy → Tensor (共享内存)
to_np = from_np.numpy()                         # Tensor → NumPy (共享内存)

# --- 1.2 张量属性 ---

x = torch.randn(3, 4, 5)
x.shape      # 形状: torch.Size([3, 4, 5])
x.size()     # 同 shape，可传参 x.size(0) → 3
x.dtype      # 数据类型: torch.float32
x.device     # 所在设备: cpu / cuda:0
x.ndim       # 维度数: 3
x.numel()    # 元素总数: 3*4*5 = 60
x.requires_grad  # 是否需要梯度（默认False）

# --- 1.3 数据类型转换 ---

x = torch.tensor([1.0, 2.0])                   # 默认 float32
x16 = x.half()               # 转为 float16 (半精度)
x32 = x.float()              # 转为 float32
x64 = x.double()             # 转为 float64
xi = x.long()                # 转为 int64
xi2 = x.int()                # 转为 int32
xb = x.bool()                # 转为布尔型
x8 = x.to(torch.float16)     # 通用转换：x.to(dtype)
# x.to(torch.device("cuda"))   # 移到GPU：x.to(device) — 需要CUDA

# --- 1.4 索引与切片 ---

x = torch.randn(4, 6)
x[0]           # 第0行
x[:, 0]        # 第0列
x[0, 0]        # 第(0,0)元素，返回标量
x[0][0]        # 同上
x[1:3, 2:5]    # 切片：第1~2行，第2~4列
x[::2, ::3]    # 步长切片：每隔2行、每隔3列取
x[x > 0]       # 布尔索引：取出所有大于0的元素（返回一维张量）

# --- 1.5 形状操作 ---

x = torch.randn(2, 3, 4)
x.view(2, 12)           # 重塑为 (2, 12)，要求与原张量元素数相同
x.view(-1, 4)           # -1 表示自动推断该维度 → (6, 4)
x.reshape(2, 12)        # reshape：更灵活，可能返回副本
x.unsqueeze(0)          # 在第0维增加一维 → (1, 2, 3, 4)
x.unsqueeze(-1)         # 在最后一维增加 → (2, 3, 4, 1)
x.squeeze()             # 去除所有大小为1的维度
x.permute(2, 0, 1)      # 维度重排 → (4, 2, 3)
x.transpose(0, 2)       # 交换第0和第2维 → (4, 3, 2)

x2d = torch.randn(3, 4)
x2d.t()                 # 二维转置（仅限2D） → (4, 3)

x.flatten()             # 展平为一维
x.flatten(start_dim=1)  # 从第1维开始展平（保留batch维）
x.repeat(2, 1, 1)       # 沿第0维重复2次 → (4, 3, 4)
x.unsqueeze(0).expand(4, -1, -1, -1)  # 扩展维度（不复制数据，仅改变stride）
# expand要求被扩展的维度大小必须为1

# --- 1.6 拼接与分割 ---

a, b = torch.randn(2, 3), torch.randn(2, 3)
cat0 = torch.cat([a, b], dim=0)      # 沿第0维拼接 → (4, 3)
cat1 = torch.cat([a, b], dim=1)      # 沿第1维拼接 → (2, 6)
stacked = torch.stack([a, b], dim=0)  # 在新维度堆叠 → (2, 2, 3)

x = torch.randn(8, 4)
chunks = torch.chunk(x, 4, dim=0)    # 沿dim=0均分成4份，每份 (2,4)
splits = torch.split(x, [3, 5], dim=0)  # 按 [3,5] 分割 → (3,4) 和 (5,4)

# --- 1.7 数学运算 ---

a, b = torch.randn(3, 4), torch.randn(3, 4)

# 逐元素运算
c = a + b              # 加法
c = a - b              # 减法
c = a * b              # 逐元素乘法（不是矩阵乘法！）
c = a / b              # 除法
c = a ** 2             # 幂运算
c = torch.add(a, b)    # 函数形式
torch.add(a, b, out=c) # 指定输出张量

# 矩阵运算
c = a @ b.T            # 矩阵乘法 (推荐写法)
c = torch.mm(a, b.T)   # 矩阵乘法（仅2D）
c = torch.matmul(a, b.T)  # 通用矩阵乘法（支持广播）
# 批量矩阵乘法需要 3D 张量 (batch, m, n) @ (batch, n, p)
c = torch.bmm(torch.randn(4, 3, 4), torch.randn(4, 4, 5))  # → (4, 3, 5)

# 归约操作
x = torch.randn(3, 4)
x.sum()                 # 所有元素求和
x.sum(dim=0)            # 沿第0维求和 → (4,)
x.sum(dim=0, keepdim=True)  # 保持维度 → (1, 4)
x.mean()                # 均值
x.max()                 # 最大值
x.min()                 # 最小值
x.argmax(dim=1)         # 沿dim=1最大值的索引
x.argmin(dim=0)         # 沿dim=0最小值的索引
x.std()                 # 标准差
x.var()                 # 方差
x.norm()                # L2范数
x.abs()                 # 绝对值
x.sqrt()                # 平方根
x.exp()                 # e^x
x.log()                 # ln(x)
x.log2()                # log2(x)
x.log10()               # log10(x)
x.clamp(min=0)          # 下界裁剪 → 相当于 ReLU
x.clamp(-0.5, 0.5)      # 上下界裁剪

# 比较操作
y = torch.randn(3, 4)
x.eq(y)                 # 逐元素 ==
x.gt(y)                 # 逐元素 >
x.lt(y)                 # 逐元素 <
x.ge(y)                 # 逐元素 >=
x.le(y)                 # 逐元素 <=
torch.allclose(x, y)    # 近似相等（容忍浮点误差）

# 统计操作
x = torch.randn(100)
x.topk(5)               # 最大的5个值及其索引
x.kthvalue(3, dim=0)    # 第3小的值

# --- 1.8 就地操作 (in-place) ---

x = torch.randn(3, 4)
x.add_(1)               # _ 后缀表示就地操作，等同 x += 1
x.mul_(2)               # x *= 2
x.zero_()               # 清零
x.fill_(3.14)           # 全部填为 3.14
x.normal_(0, 1)         # 用 N(0,1) 填充

# --- 1.9 设备转换 ---

x = torch.randn(3, 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)                # 通用写法
# x = x.cuda()                    # 直接移到 GPU（需要 CUDA 可用）
# x = x.cpu()                     # 移回 CPU

# =============================================================================
# 二、自动求导 (Autograd)
# =============================================================================

# --- 2.1 基本用法 ---

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.pow(2).sum()              # y = x[0]^2 + x[1]^2
y.backward()                    # 反向传播，计算梯度
print(x.grad)                   # dy/dx = [4.0, 6.0]

# --- 2.2 多步梯度计算 ---

x = torch.randn(3, requires_grad=True)
y = x * 2                       # y = 2x
z = y.sum()
z.backward()
print(x.grad)                   # dz/dx = [2, 2, 2]

# --- 2.3 梯度清零 ---

x = torch.randn(3, requires_grad=True)
for i in range(5):
    y = (x * 1).sum()
    y.backward()
    # x.grad 会累加！每次 backward 梯度会叠加
x.grad.zero_()                  # 手动清零（训练循环中必须做）
# 或使用: optimizer.zero_grad()

# --- 2.4 不需要梯度时 ---

# 方法1：创建时就设置
x = torch.randn(3, requires_grad=False)

# 方法2：上下文管理器
with torch.no_grad():
    y = x * 2                   # 此区域内不追踪梯度（推理/评估时使用）

# 方法3：原地关闭
x.requires_grad_(False)

# --- 2.5 获取梯度计算细节 ---

x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()
z.backward()
print(x.is_leaf)                # True, x 是用户创建的叶子节点
print(y.is_leaf)                # False, y 由运算产生
print(x.grad)                   # 梯度值
print(x.grad_fn)                # None（叶子节点无grad_fn）
print(y.grad_fn)                # <MulBackward0> 产生y的运算

# --- 2.6 带标量的 backward ---

x = torch.randn(3, 4, requires_grad=True)
y = x.sum(dim=1)                # y 形状为 (3,)
# y.backward()                  # 报错！y不是标量
y.backward(torch.ones_like(y))  # 传入与y同形状的梯度

# =============================================================================
# 三、神经网络层 (nn.Module)
# =============================================================================

# --- 3.1 全连接层 ---

# nn.Linear(in_features, out_features, bias=True)
linear = nn.Linear(128, 64)     # 输入128维 → 输出64维
x = torch.randn(32, 128)        # batch=32, features=128
out = linear(x)                 # 输出形状 (32, 64)
# 内部执行: out = x @ W.T + b

# --- 3.2 卷积层 ---

# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
x = torch.randn(8, 3, 32, 32)   # (batch, 通道, 高, 宽)
out = conv(x)                   # 输出形状 (8, 16, 32, 32)
# 输出尺寸公式: H_out = (H_in + 2*padding - kernel_size) / stride + 1

# nn.Conv1d — 一维卷积（时序数据）
conv1d = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
x1d = torch.randn(4, 1, 100)    # (batch, channels, length)
out1d = conv1d(x1d)             # (4, 8, 98)

# nn.ConvTranspose2d — 转置卷积（上采样）
deconv = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

# --- 3.3 池化层 ---

# 最大池化
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# 自适应池化（强制输出指定尺寸）
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化 → (B, C, 1, 1)

x = torch.randn(8, 16, 32, 32)
out = maxpool(x)                # (8, 16, 16, 16)
out = adaptive_pool(x)          # (8, 16, 1, 1)

# --- 3.4 归一化层 ---

# 批归一化（2D — 适用于卷积输出）
bn = nn.BatchNorm2d(16)         # 16 是通道数

# 层归一化（适用于 Transformer/RNN）
ln = nn.LayerNorm(normalized_shape=128)

# 实例归一化（风格迁移常用）
inst = nn.InstanceNorm2d(16)

# 组归一化
gn = nn.GroupNorm(num_groups=4, num_channels=16)

# --- 3.5 循环神经网络层 ---

# nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
rnn = nn.RNN(128, 256, num_layers=2, batch_first=True)
x = torch.randn(32, 10, 128)    # (batch, seq_len, input_size)
out, hn = rnn(x)                # out: (32,10,256), hn: (2,32,256)

# nn.LSTM — 长短时记忆网络
lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, bidirectional=True)
out, (hn, cn) = lstm(x)         # out: (32,10,512) 双向拼接

# nn.GRU — 门控循环单元（LSTM简化版，参数更少）
gru = nn.GRU(128, 256, num_layers=2, batch_first=True)

# --- 3.6 激活函数（类形式，用于 Sequential） ---

relu = nn.ReLU()                # ReLU: max(0, x)
relu6 = nn.ReLU6()              # 裁剪版ReLU: min(max(0,x), 6)
lrelu = nn.LeakyReLU(0.01)      # LeakyReLU: x>0时x，否则0.01x
prelu = nn.PReLU()              # 参数化ReLU，负斜率可学习
elu = nn.ELU()                  # ELU: x>0时x，否则α(e^x-1)
selu = nn.SELU()                # SELU: 自归一化ELU
gelu = nn.GELU()                # GELU: 高斯误差线性单元（Transformer常用）
sigmoid = nn.Sigmoid()          # 1 / (1+e^-x) → (0, 1)
tanh = nn.Tanh()                # tanh → (-1, 1)
softmax = nn.Softmax(dim=1)     # Softmax（多分类输出层）
log_softmax = nn.LogSoftmax(dim=1)  # LogSoftmax（配合 NLLLoss）
softplus = nn.Softplus()        # ln(1+e^x)，ReLU的平滑近似

# --- 3.7 Dropout 层 ---

dropout = nn.Dropout(p=0.5)     # 随机将50%的神经元置0（训练时生效）
drop2d = nn.Dropout2d(p=0.2)    # 按通道丢弃（卷积用）

# --- 3.8 Embedding 层 ---

# nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
embedding = nn.Embedding(10000, 256, padding_idx=0)  # 词汇量10000，嵌入维度256
indices = torch.randint(0, 10000, (32, 50))           # (batch, seq_len)
out = embedding(indices)       # (32, 50, 256)

# --- 3.9 Transformer 层 ---

# nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, ...)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True
)
# 更完整的用法见 torch.nn.Transformer

# --- 3.10 Flatten 层 ---

flatten = nn.Flatten()          # 将 (B, C, H, W) → (B, C*H*W)

# =============================================================================
# 四、函数式 API (torch.nn.functional) — 常用缩写为 F
# =============================================================================

x = torch.randn(32, 128)

# 激活函数（函数式，无状态）
out = F.relu(x)
out = F.leaky_relu(x, negative_slope=0.01)
out = F.gelu(x)
out = F.sigmoid(x)
out = F.tanh(x)
out = F.softmax(x, dim=1)
out = F.log_softmax(x, dim=1)
out = F.softplus(x)
out = F.elu(x)

# 函数式 Dropout（训练时需要传 training=True）
out = F.dropout(x, p=0.5, training=True)

# 函数式归一化（F.batch_norm需要传入running_mean/running_var，一般直接用nn.BatchNorm层）
out = F.layer_norm(x, normalized_shape=(128,))

# 函数式卷积/池化（常用于自定义 forward，需要传入weight/bias参数）
# 用法: F.conv2d(input, weight, bias, stride=1, padding=0)
# 用法: F.max_pool2d(input, kernel_size)
# 用法: F.avg_pool2d(input, kernel_size)
# 用法: F.adaptive_avg_pool2d(input, output_size)

# 独热编码
out = F.one_hot(torch.tensor([0, 2, 1]), num_classes=3)
# → tensor([[1,0,0], [0,0,1], [0,1,0]])

# 交叉熵损失（函数式中直接可用）
# loss = F.cross_entropy(logits, labels)  # 内部包含 log_softmax + nll_loss

# =============================================================================
# 五、构建与训练神经网络
# =============================================================================

# --- 5.1 自定义 nn.Module ---

class SimpleMLP(nn.Module):
    """一个简单的多层感知机"""
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # 在 __init__ 中定义所有子模块
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)      # 批归一化
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        """定义前向传播"""
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # 返回 logits，不在这里做 softmax

# --- 5.2 使用 nn.Sequential 快速构建 ---

conv_net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),    # (B, 3, H, W) → (B, 32, H, W)
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),                     # 下采样到 1/2
    nn.Conv2d(32, 64, 3, padding=1),   # (B, 64, H/2, W/2)
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),                     # 下采样到 1/4
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 256),         # 假设输入 32×32
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
)

# --- 5.3 参数初始化 ---

def init_weights(m):
    """自定义权重初始化函数"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model = SimpleMLP(784, 256, 10)
model.apply(init_weights)               # 递归地对所有子模块执行 init_weights

# 常用初始化方法
# nn.init.xavier_uniform_(tensor)      — Xavier均匀（sigmoid/tanh 适用）
# nn.init.kaiming_normal_(tensor)      — Kaiming正态（ReLU 适用）
# nn.init.normal_(tensor, 0, 0.02)     — 自定义正态
# nn.init.constant_(tensor, val)       — 常量
# nn.init.zeros_(tensor)               — 全零
# nn.init.ones_(tensor)                — 全一

# --- 5.4 模型信息 ---

print(model)                          # 打印模型结构
for name, param in model.named_parameters():    # 遍历所有参数
    print(name, param.shape, param.requires_grad)
for name, buf in model.named_buffers():          # 遍历buffer（如BN的running_mean）
    print(name, buf.shape)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# --- 5.5 模型模式切换 ---

model.train()                         # 训练模式（启用 Dropout、BatchNorm 更新统计量）
model.eval()                          # 评估模式（冻结 Dropout、BatchNorm 使用移动平均值）

# =============================================================================
# 六、损失函数 (Loss Functions)
# =============================================================================

# --- 6.1 分类损失 ---

# 交叉熵损失 — 最常用分类损失
# nn.CrossEntropyLoss() = LogSoftmax + NLLLoss
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)          # (batch, num_classes)，未经过softmax
labels = torch.randint(0, 10, (32,))  # (batch,)，类别索引（不是one-hot！）
loss = ce_loss(logits, labels)        # 自动做 softmax

# 带权重的交叉熵（处理类别不平衡）
weights = torch.tensor([1.0, 2.0, 1.5])  # 各类别权重
ce_weighted = nn.CrossEntropyLoss(weight=weights)

# 二元交叉熵 — 二分类
bce_loss = nn.BCEWithLogitsLoss()     # 包含 sigmoid + BCE（推荐，数值稳定）
logits = torch.randn(32, 1)           # (batch, 1)
labels = torch.randint(0, 2, (32, 1)).float()
loss = bce_loss(logits, labels)

# NLLLoss — 配合 LogSoftmax 使用
nll = nn.NLLLoss()

# --- 6.2 回归损失 ---

# 均方误差
mse = nn.MSELoss()
# loss = mse(pred, target)

# 平均绝对误差 (L1 Loss)
l1 = nn.L1Loss()
# loss = l1(pred, target)

# Smooth L1 Loss — L1和MSE的结合（目标检测常用，对离群点不敏感）
smooth_l1 = nn.SmoothL1Loss(beta=1.0)

# Huber Loss — Smooth L1的变体
huber = nn.HuberLoss(delta=1.0)

# --- 6.3 其他损失 ---

# KL散度 — 衡量两个分布的差异
kl = nn.KLDivLoss(reduction='batchmean')  # reduction='batchmean' 是数学上正确的做法

# 余弦相似度损失 — 使向量方向一致
cosine = nn.CosineEmbeddingLoss()

# Margin Ranking Loss — 排序学习
margin_rank = nn.MarginRankingLoss(margin=1.0)

# Triplet Margin Loss — 三元组损失（人脸识别/度量学习）
triplet = nn.TripletMarginLoss(margin=1.0, p=2.0)

# CTC Loss — 时序分类（语音识别）
# ctc = nn.CTCLoss(blank=0, zero_infinity=True)

# --- 6.4 自定义损失函数 ---

class CustomLoss(nn.Module):
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        l1_reg = pred.abs().mean() * 0.001  # L1正则项
        return mse_loss + l1_reg

# =============================================================================
# 七、优化器与学习率调度
# =============================================================================

model = SimpleMLP(784, 256, 10)

# --- 7.1 优化器 ---

# SGD — 随机梯度下降
# optim.SGD(params, lr, momentum=0, weight_decay=0, nesterov=False)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam — 自适应学习率优化器（最常用）
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

# AdamW — Adam + 解耦权重衰减（Transformer训练推荐）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop — 适合RNN/强化学习
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Adagrad — 自适应梯度（适合稀疏数据）
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# LBFGS — 准牛顿法（小数据集微调用，闭包形式）
# optimizer = optim.LBFGS(model.parameters(), lr=1)

# --- 7.2 学习率调度器 ---

scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# 每 step_size 个 epoch 将 lr 乘以 gamma

scheduler_multisteplr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
# 在第30、80个epoch将lr乘以gamma

scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# 每个epoch将 lr 乘以 gamma

scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# 余弦退火：lr 沿余弦曲线降到0

scheduler_cosine_warm = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
# 余弦退火+热重启（训练更长时间时有效）

scheduler_reduce = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
# 当指标不再改善时降低lr（用法见训练循环）

scheduler_onecycle = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=100, epochs=10, cycle_momentum=False)
# cycle_momentum=False: 当优化器不支持momentum时需设为False
# OneCycle 策略（先升后降）

# --- 7.3 完整的训练步骤 ---

# for epoch in range(num_epochs):
#     # 训练阶段
#     model.train()
#     for x_batch, y_batch in train_loader:
#         x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#
#         optimizer.zero_grad()       # 1. 清零梯度
#         output = model(x_batch)     # 2. 前向传播
#         loss = ce_loss(output, y_batch)  # 3. 计算损失
#         loss.backward()             # 4. 反向传播
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪（可选）
#         optimizer.step()            # 5. 更新参数
#
#     # 评估阶段
#     model.eval()
#     with torch.no_grad():           # 不计算梯度
#         for x_batch, y_batch in val_loader:
#             output = model(x_batch)
#             val_loss = ce_loss(output, y_batch)
#
#     scheduler.step()                # 6. 更新学习率
#     # 如果是 ReduceLROnPlateau: scheduler.step(val_loss)

# =============================================================================
# 八、数据加载 (Dataset & DataLoader)
# =============================================================================

# --- 8.1 使用 TensorDataset 直接加载 ---

x_data = torch.randn(1000, 28, 28)   # 1000 张 28×28 的"图片"
y_data = torch.randint(0, 10, (1000,))
dataset = TensorDataset(x_data, y_data)

loader = DataLoader(
    dataset,
    batch_size=64,       # 每批 64 个样本
    shuffle=True,        # 每个 epoch 随机打乱
    num_workers=0,       # 子进程数（Windows 设为 0，Linux 可设 >0）
    drop_last=True,      # 丢弃最后不足 batch_size 的批次
    pin_memory=True,     # 锁页内存（GPU训练时加速传输）
)
for x_batch, y_batch in loader:
    print(x_batch.shape, y_batch.shape)  # (64, 28, 28) 和 (64,)

# --- 8.2 自定义 Dataset ---

class CustomDataset(Dataset):
    """自定义数据集模板"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """返回数据集总大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """返回第 idx 个样本"""
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# --- 8.3 数据集划分 ---

dataset = TensorDataset(torch.randn(1000, 28, 28), torch.randint(0, 10, (1000,)))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# --- 8.4 torchvision 预置数据集 ---

# MNIST — 取消注释以下载:
# train_data = datasets.MNIST(root='./data', train=True, download=True,
#                              transform=transforms.ToTensor())
# CIFAR10 — 取消注释以下载:
# cifar10 = datasets.CIFAR10(root='./data', train=True, download=True,
#                             transform=transforms.ToTensor())
# ImageNet（需要先下载到本地）: datasets.ImageNet(root='./data', split='train')

# --- 8.5 常用数据增强 (transforms) ---

# compose 可以组合多个 transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),          # 缩放到 256×256
    transforms.RandomResizedCrop(224),       # 随机裁剪到 224×224
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.RandomVerticalFlip(p=0.5),    # 50%概率垂直翻转
    transforms.RandomRotation(degrees=15),   # 随机旋转 ±15 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),       # 10%概率转灰度
    transforms.GaussianBlur(kernel_size=3),  # 高斯模糊
    transforms.ToTensor(),                   # PIL Image / numpy → Tensor [0,1]
    transforms.Normalize(                    # 标准化
        mean=[0.485, 0.456, 0.406],         # ImageNet 均值
        std=[0.229, 0.224, 0.225]           # ImageNet 标准差
    ),
])

# =============================================================================
# 九、GPU 训练
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 9.1 单GPU ---

model = SimpleMLP(784, 256, 10).to(device)   # 模型移到 GPU
criterion = nn.CrossEntropyLoss()

for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)              # 数据移到 GPU
    y_batch = y_batch.to(device)

    optimizer.zero_grad()
    output = model(x_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    optimizer.step()

# --- 9.2 多GPU (DataParallel — 简单但不推荐用于生产) ---

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)            # 自动分配到多GPU
# 注意：DataParallel 存在负载不均衡问题

# --- 9.3 分布式数据并行 (DistributedDataParallel — 推荐) ---
# DDP 通常通过 torchrun 启动脚本：
# python -m torch.distributed.run --nproc_per_node=4 train.py

# import torch.distributed as dist
# dist.init_process_group(backend='nccl')
# model = model.to(device)
# model = nn.parallel.DistributedDataParallel(model)
# 配合 DistributedSampler 使用

# --- 9.4 混合精度训练 (AMP — Automatic Mixed Precision) ---

# 混合精度训练示例（需要 CUDA）
# scaler = torch.amp.GradScaler('cuda')           # PyTorch 2.0+ 推荐写法
# for x_batch, y_batch in train_loader:
#     x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#     optimizer.zero_grad()
#     with torch.amp.autocast('cuda'):            # 自动混合精度上下文
#         output = model(x_batch)
#         loss = criterion(output, y_batch)
#     scaler.scale(loss).backward()               # loss * scale 后反向传播
#     scaler.step(optimizer)                      # 更新参数（自动 unscale）
#     scaler.update()                             # 更新 scale 因子
# # 优势：减少显存，加速训练

# --- 9.5 查看GPU信息 ---

print(f"GPU可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"显存分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"显存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# =============================================================================
# 十、模型保存与加载
# =============================================================================

model = SimpleMLP(784, 256, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch = 10
loss_value = 0.123

# --- 10.1 保存完整 checkpoint ---

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),       # 模型参数
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
    'loss': loss_value,
}
torch.save(checkpoint, 'checkpoint.pth')           # 保存为 .pth 或 .pt 文件

# --- 10.2 加载 checkpoint ---

checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
last_loss = checkpoint['loss']

# --- 10.3 仅保存/加载模型参数（推荐用于部署） ---

torch.save(model.state_dict(), 'model_weights.pth')  # 只保存参数
model.load_state_dict(torch.load('model_weights.pth', map_location=device))  # 加载参数

# --- 10.4 保存/加载整个模型（不推荐，依赖文件结构） ---

# torch.save(model, 'model_full.pth')
# model = torch.load('model_full.pth')

# --- 10.5 导出为 TorchScript（跨语言部署） ---

scripted = torch.jit.script(model)                # trace或script方式
scripted.save('model_scripted.pt')
loaded_model = torch.jit.load('model_scripted.pt')

# =============================================================================
# 十一、可视化与调试
# =============================================================================

# --- 11.1 TensorBoard 集成 ---

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[INFO] TensorBoard 未安装，跳过 TensorBoard 示例。安装: pip install tensorboard")

# TensorBoard 示例（需要: pip install tensorboard）
if HAS_TENSORBOARD:
    writer = SummaryWriter('runs/experiment_1')

    for epoch in range(10):
        train_loss = 0.9 - epoch * 0.05
        val_loss = 1.0 - epoch * 0.03
        acc = 0.6 + epoch * 0.04

        writer.add_scalar('Loss/train', train_loss, epoch)  # 标量曲线
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

    writer.add_graph(model, torch.randn(1, 784))      # 模型结构图
    writer.add_embedding(torch.randn(100, 64))        # 降维可视化
    writer.close()
# 启动: tensorboard --logdir=runs

# --- 11.2 模型参数统计 ---

# from torchsummary import summary                  # pip install torchsummary
# summary(model, input_size=(784,), device='cpu')  # 模型结构摘要

# 或用 PyTorch 内置：
def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# --- 11.3 梯度检查 ---

def check_grad(model):
    """检查梯度是否正常"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 100:
                print(f"[WARN] 梯度爆炸 {name}: {grad_norm:.2f}")
            elif grad_norm < 1e-7 and param.requires_grad:
                print(f"[WARN] 梯度消失 {name}: {grad_norm:.7f}")

# --- 11.4 模型 FLOPs ---
# pip install thop
# from thop import profile
# flops, params = profile(model, inputs=(torch.randn(1, 784),))
# print(f"FLOPs: {flops / 1e6:.2f}M, Params: {params / 1e6:.2f}M")

# =============================================================================
# 十二、预训练模型与迁移学习
# =============================================================================

# --- 12.1 加载 torchvision 预训练模型 ---

# 常用模型：resnet18/34/50/101/152, vgg16/19, densenet121, mobilenet_v2, efficientnet_b0, vit_b_16
model = models.resnet18(pretrained=True)           # 加载 ImageNet 预训练权重

# --- 12.2 冻结特征提取层 ---

for param in model.parameters():
    param.requires_grad = False                     # 冻结所有层

# 替换最后一层
num_features = model.fc.in_features                 # ResNet: 512
model.fc = nn.Linear(num_features, 10)              # 改为10分类

# 只训练最后一层
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# --- 12.3 分层设置学习率 (微调) ---

optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},  # 浅层用更小lr
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},     # 新层用更大lr
])

# =============================================================================
# 十三、常用工具函数
# =============================================================================

# --- 13.1 设置随机种子（保证可复现） ---

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True   # 最大确定性（可能变慢）
    # torch.backends.cudnn.benchmark = False      # 关闭自动调优

set_seed(42)

# --- 13.2 梯度裁剪 ---

# 梯度裁剪示例（需先做 backward 才有梯度可裁剪）
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 按范数裁剪
# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)  # 按值裁剪

# --- 13.3 获取预测结果 ---

def predict(model, x, device='cpu', return_probs=True):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)          # 转为概率
        preds = torch.argmax(probs, dim=1)        # 预测类别
        return (probs, preds) if return_probs else preds

# --- 13.4 准确率计算 ---

def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k / batch_size).item())
        return res

# top1_acc, top5_acc = accuracy(output, labels, topk=(1, 5))
# print(f"Top-1: {top1_acc:.2%}, Top-5: {top5_acc:.2%}")

# --- 13.5 学习率查找器（简化版） ---

def find_lr(model, optimizer, criterion, train_loader, device, start_lr=1e-7, end_lr=10, num_iter=100):
    """快速学习率查找"""
    model.train()
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    losses, lrs = [], []
    it = iter(train_loader)
    best_loss = float('inf')

    for i in range(num_iter):
        try:
            x_batch, y_batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            x_batch, y_batch = next(it)

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if loss.item() < best_loss:
            best_loss = loss.item()
        elif loss.item() > best_loss * 4:  # loss爆炸则停止
            break

    return lrs, losses

# --- 13.6 保存和加载训练的 epoch ---

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler_cosine.state_dict(),
    'loss': loss,
}, 'checkpoint_full.pth')

# --- 13.7 批量矩阵乘法示例 ---

# 当有多个独立的矩阵乘法时，使用 torch.einsum 更简洁
a = torch.randn(4, 5, 6)             # 4个 5×6 矩阵
b = torch.randn(4, 6, 7)             # 4个 6×7 矩阵
c = torch.einsum('bij,bjk->bik', a, b)  # (4, 5, 7) — 批量矩阵乘法
# 等价于 torch.bmm(a, b)

# --- 13.8 条件判断 ---

def check_nan(model):
    """检查模型参数是否存在 NaN"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in {name}")
            return True
    return False

# =============================================================================
# 十四、完整训练脚本模板
# =============================================================================

def train_and_evaluate(
    model, train_loader, val_loader, criterion, optimizer,
    scheduler, device, num_epochs=50, checkpoint_path='best_model.pth'
):
    """完整的训练和验证流程"""
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ========== 训练 ==========
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()                    # 清零梯度
            outputs = model(x_batch)                 # 前向传播
            loss = criterion(outputs, y_batch)       # 计算损失
            loss.backward()                          # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()                         # 参数更新

            train_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        # ========== 验证 ==========
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        # 更新学习率
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [SAVED] 保存最佳模型 (acc: {val_acc:.4f})")

        # 打印进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
    return best_val_acc

# --- 使用示例 ---
# model = SimpleMLP(784, 256, 10).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# criterion = nn.CrossEntropyLoss()
# train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

# =============================================================================
# 十五、PyTorch 常用代码片段速查
# =============================================================================

# -- 张量创建 --
# torch.tensor(data)         从数据创建
# torch.zeros/ones/randn/rand/randint/eye/full  特殊张量
# torch.arange/linspace/logspace                   序列张量

# -- 设备操作 --
# x.to(device)              移到指定设备
# x.cuda() / x.cpu()        移到GPU/CPU

# -- 自动求导 --
# x.requires_grad_(True)    启用梯度
# loss.backward()           反向传播
# optimizer.zero_grad()     清零梯度

# -- 模型操作 --
# model.train() / model.eval()    切换模式
# model.to(device)                模型移到设备
# model.apply(fn)                 递归应用函数
# model.state_dict()              获取参数字典

# -- 保存/加载 --
# torch.save(obj, path)          保存
# model.load_state_dict(torch.load(path))  加载参数

# -- 常用层 --
# nn.Linear / nn.Conv2d / nn.BatchNorm2d / nn.Dropout / nn.ReLU
# nn.MaxPool2d / nn.AdaptiveAvgPool2d / nn.Embedding / nn.Flatten

# -- 常用损失 --
# nn.CrossEntropyLoss / nn.MSELoss / nn.BCEWithLogitsLoss / nn.L1Loss

# -- 常用优化器 --
# optim.SGD / optim.Adam / optim.AdamW / optim.RMSprop

# -- 上下文管理器 --
# torch.no_grad()            不计算梯度
# torch.cuda.amp.autocast()  混合精度

print("PyTorch 指南加载完成！")
print(f"当前设备: {device}")
print(f"PyTorch 版本: {torch.__version__}")
