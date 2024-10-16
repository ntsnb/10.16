import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import torch.nn.init as init
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# =====================================================================================分别加载两个数据集===================================================================================
def Load_data(Dataset):
    edges = []
    with open(Dataset+'/A.txt', 'r') as f:
        for line in f:
            try:
                # 去除换行符并解析 x 和 y
                x, y = map(int, line.strip().split(','))
                edges.append((x, y))
            except ValueError:
                # 如果出现解析错误，可以忽略该行
                continue

    # 将解析后的 edges 转换为 NumPy 数组
    edges = np.array(edges).T  # 转置为 (2, num_edges) 的形状
    # 将 NumPy 数组转换为 PyTorch LongTensor
    edge_index = torch.tensor(edges, dtype=torch.long)

    # val_idx = np.load('politifact/train_idx.npy')  # 代表验证集图的ID
    node_graph_id = np.load(Dataset+'/node_graph_id.npy')  # 每个节点所属的图的ID
    npz_file = np.load(Dataset+'/new_content_feature.npz')
    graph_labels = np.load(Dataset+'/graph_labels.npy')  # 图的标签

    data = npz_file['data']           # 非零元素的值
    indices = npz_file['indices']     # 列索引
    indptr = npz_file['indptr']       # 行指针
    shape = tuple(npz_file['shape'])  # 特征矩阵的形状 (num_nodes, num_features)

    # 3. 构建 CSR 稀疏矩阵
    sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
    dense_matrix = sparse_matrix.toarray()

    batch = torch.tensor(node_graph_id, dtype=torch.long)

    node_features = dense_matrix
    x = torch.tensor(node_features, dtype=torch.float)
    graph_labels = graph_labels
    y = torch.tensor(graph_labels, dtype=torch.long)
    print(f"提取出的节点特征矩阵形状: {x.shape}")
    print(f"提取出的图标签形状: {y.shape}")

    data = Data(x=x, edge_index=edge_index, y=y, batch=batch)

    data_list = []

    # 遍历每个图，确保正确筛选每张图的节点和边
    cont = 0
    for i in torch.unique(data.batch):  # 遍历 546 个图
        # 找到属于图 i 的节点（通过 batch 向量）
        mask = data.batch == i
        node_indices = mask.nonzero(as_tuple=True)[0]  # 当前图的节点索引
        print(node_indices)

        # 筛选出与当前图相关的边：两端点都在图 i 中
        edge_mask = (mask[data.edge_index[0]] & mask[data.edge_index[1]])
        sub_edge_index = data.edge_index[:, edge_mask]
        print(f"sub_edge_index:{sub_edge_index}")

        # 构建新的 `edge_index`，确保索引重新编号从 0 开始
        mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        print(mapping)
        new_edge_index = torch.tensor(
            [[mapping[old.item()] for old in sub_edge_index[0]],
            [mapping[old1.item()] for old1 in sub_edge_index[1]]],
            dtype=torch.long
        )


        # 构建新的 Data 对象
        new_data = Data(
            x=data.x[mask],  # 当前图的节点特征
            edge_index=new_edge_index,  # 当前图的边索引
            y=data.y[cont].unsqueeze(0)  # 当前图的标签
        )
        new_data.dataset = Dataset # 标记数据集来源
        cont += 1
        print(f"i,{i}")
        data_list.append(new_data)
    input_dim = x.shape[1]
    output_dim = y.max().item() + 1  # 假设 y 是分类标签，输出维度等于类别数量
    return data_list, input_dim, output_dim

data_list1, input_dim1, output_dim1 = Load_data('gossipcop')
data_list2, input_dim2, output_dim2 = Load_data('politifact')
if input_dim2 == input_dim1 and output_dim1 == output_dim2:
    print("两数据集可以联合训练")


random.seed(2)
# =================================自己设置保证数据集平衡，因为gossipcop数据偏多===============
num = 314
data_list1 = random.sample(data_list1, num)
data_list = data_list1 + data_list2
train_loader = DataLoader(data_list, batch_size=32, shuffle=True)

class GATGraphClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=7, heads=1, dropout=0.5):
        """
        两层 GAT 的图分类模型，支持 Dropout。

        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 每个头的隐藏层维度
            output_dim (int): 输出类别数量
            heads (int): 注意力头的数量
            dropout (float): Dropout 概率
        """
        super(GATGraphClassifier, self).__init__()
        # 第一层 GAT：输入 -> 隐藏层维度 * 注意力头数
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        
        # 第二层 GAT：多头聚合后 -> 输出
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        # 全连接层用于分类
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重"""
        init.kaiming_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 第一层 GAT + ReLU + Dropout
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层 GAT + Dropout
        x = self.gat2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 全局池化，将节点特征聚合为图特征
        x = global_mean_pool(x, batch)
        self.features = x

        # 全连接层用于分类
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

input_dim = input_dim1
hidden_dim = 64  # 可以根据需要调整
output_dim = output_dim1  

# model = GCNGraphClassifier(input_dim, hidden_dim, output_dim)
# 选择 Adam 作为优化器

model = GATGraphClassifier(
    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, heads=4, dropout=0.3
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数为交叉熵损失
criterion = torch.nn.CrossEntropyLoss()



# 将模型设置为训练模式
model.train()

# 训练模型
for epoch in range(200):  # 迭代 200 次
    for batch in train_loader:  # 每次加载一个 batch 的图
        optimizer.zero_grad()  # 清空梯度
        out = model(batch)  # 前向传播，batch 包含多个图
        loss = criterion(out, batch.y)  # 计算损失，batch.y 是每个图的标签
        loss.backward()  # 反向传播
        optimizer.step()  # 优化模型
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 保存模型权重
torch.save(model.state_dict(), "gat_model_weights9 2 1.pth")
print("模型训练完成并已保存权重")

# 训练完成后，可以将模型设置为评估模式
model.load_state_dict(torch.load("gat_model_weights9 2 1.pth"))
model.eval()
correct = 0
for batch in train_loader:
    out = model(batch)
    pred = out.argmax(dim=1)
    correct += (pred == batch.y).sum().item()

accuracy = correct / len(data_list)
print(f'Accuracy: {accuracy:.4f}')

#===================作图====================
features = []
labels = []
colors = []  # 用于存储每个数据集的标签，用不同颜色区分
idx_1 = []
idx_2 = []
cont = 0

# 遍历数据集并提取特征
for batch in train_loader:
    out = model(batch)  # 前向传播，计算特征
    features.append(model.features.detach().cpu().numpy())  # 提取全连接层前的特征
    labels.extend(batch.y.cpu().numpy())  # 存储标签

     # 根据每个图的 dataset 属性分配颜色
    for graph in batch.to_data_list():
        colors.append('blue' if graph.dataset == 'gossipcop' else 'red')
        if graph.dataset == 'gossipcop':
            idx_1.append(cont)
        else:
            idx_2.append(cont)
        cont+=1



# 转换特征为 numpy 数组
features = np.vstack(features)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

# 绘制 t-SNE 可视化图
plt.figure(figsize=(10, 7))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6)
plt.title('t-SNE Visualization of Features')
plt.show()

#=============================


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建数据
x1 = features_2d[idx_2, 0]  # X轴数据
y1 = features_2d[idx_2, 1]               # 第一层 PolitiFact 数据
x2 = features_2d[idx_1, 0]
y2 = features_2d[idx_1, 1]               # 第二层 GossipCop 数据
x3 = features_2d[:, 0]
y3 = features_2d[:, 1]   # 第三层 PolitiFact 和 GossipCop 的交集

# 创建一个 3D 图形对象
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制第一层 XY 平面 (z=0) - PolitiFact
ax.scatter(x1, y1, zs=0, zdir='z', label='Layer 1: PolitiFact', color='b', marker='o')

# 绘制第二层 XY 平面 (z=1) - GossipCop
ax.scatter(x2, y2, zs=1, zdir='z', label='Layer 2: GossipCop', color='r', marker='x')

# 绘制第三层 XY 平面 (z=2) - 交集
ax.scatter(x3, y3, zs=2, zdir='z', label='Layer 3: Intersection', color='g', marker='^')

# 设置轴标签和标题
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Layered Feature Plot')

# 设置 Z 轴刻度和标签
ax.set_zticks([0, 1, 2])
ax.set_zticklabels(['PolitiFact', 'GossipCop', 'Intersection'])

# 显示图例
ax.legend()

# 显示图形
plt.show()






    