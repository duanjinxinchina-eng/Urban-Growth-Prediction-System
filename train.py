import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
import time
import glob
import os
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import prepare_adj, prepare_data, add_richment
from model import HspGAT

# ===============训练=====================

# =====================训练数据============================
loader = np.load(r"./data/adjacency_matrix.npz")  # 邻接矩阵
adj_coo, adj = prepare_adj(loader)
file = pd.read_csv(r"data.csv")
data = pd.DataFrame(file)
ori = data['ori']
data = add_richment(data, adj_coo, ori)
data, ori, target = prepare_data(data)

# ====================测试数据=============================
test_loader = np.load(r"adjacency_matrix_test.npz")  # 邻接矩阵
test_adj_coo, test_adj = prepare_adj(test_loader)
test_file = pd.read_csv(r"data_test.csv")
test_data = pd.DataFrame(test_file)
test_ori = test_data['ori']
test_data = add_richment(test_data, test_adj_coo, test_ori)
test_data, test_ori, test_target = prepare_data(test_data)

rows = data.shape[0]
idx_train = range(int(rows * 0.7))
idx_val = range(int(rows * 0.7), rows)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)

# ====================模型构建==============================
in_dim = data.shape[1]
nclass = int(ori.max()) + 1
model = HspGAT(in_dim, nclass, dropout=0.6, nheads=19)


# ====================参数配置=================
import argparse
class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Model Training Configuration')

        parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
        parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')

        # Parse arguments from command line or provide empty list to simulate defaults
        self.args = parser.parse_args(args=[])

        # Add cuda flag manually
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

    def __getattr__(self, item):
        return getattr(self.args, item)


args = Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ====================训练配置==============================
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
weights = torch.tensor([1, 1, 1, 10], dtype=torch.float32)


# ====================gat精度函数==============================
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).int().sum()
    return correct / len(labels)


# ====================数据转移到 CUDA =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.cuda:
    model = model.to(device)
    data = data.to(device)
    adj = adj.to(device)
    ori = ori.to(device)
    target = target.to(device)
    test_data = test_data.to(device)
    test_adj = test_adj.to(device)
    test_ori = test_ori.to(device)
    test_target = test_target.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
data, adj, ori, target = Variable(data), Variable(adj), Variable(ori), Variable(target) # Variable变量参与反向传播


# ====================训练函数==============================
def train(epoch):
    t = time.time()
    model.train()
    output = model(data, adj)
    loss_train = criterion(output[idx_train], ori[idx_train]) # CrossEntropyLoss
    acc_train = accuracy(output[idx_train], ori[idx_train]) # GAT精度
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(data, adj)

    loss_val = criterion(output[idx_val], ori[idx_val])
    acc_val = accuracy(output[idx_val], ori[idx_val])

    print(f'Epoch: {epoch + 1:04d} | '
          f'loss_train: {loss_train.item():.4f} | '
          f'acc_train: {acc_train.item():.4f} | '
          f'loss_val: {loss_val.item():.4f} | '
          f'acc_val: {acc_val.item():.4f} | '
          f'time: {time.time() - t:.4f}s')

    return loss_val.item()


# ====================测试函数==============================
def compute_test():
    model.eval()
    output = model(test_data, test_adj)
    loss_test = criterion(output, test_ori)
    acc_test = accuracy(output, test_ori)
    print(f"Test set results: "
          f"loss= {loss_test.item():.4f} | "
          f"accuracy= {acc_test.item():.4f}")


# ====================训练过程==============================
t_total = time.time()
epochs = 500
loss_values = []
best = epochs + 1
best_epoch = 0
bad_counter = 0

for epoch in range(epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), f'{epoch}.pkl')

    if loss_values[-1] < best:

        # 获取当前的注意力系数,并保存
        attention_weights = model.best_attention.coalesce()  # 标准化
        indices = attention_weights.indices().cpu().numpy()  # shape [2, E]
        values = attention_weights.values().cpu().numpy()  # shape [E]
        shape = attention_weights.shape  # tuple (N, N)
        row = indices[0]
        col = indices[1]
        sparse_matrix = sp.coo_matrix((values, (row, col)), shape=shape)
        sp.save_npz("attention_weights.npz", sparse_matrix)

        # 更新值
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0

    else:
        bad_counter += 1

    if bad_counter == args.patience:
        print("Early stopping triggered.")
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

# 保留最佳模型
files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb != best_epoch:
        os.remove(file)

print("Optimization Finished!")
print(f"Total time elapsed: {time.time() - t_total:.4f}s")

print(f'Loading {best_epoch}th epoch')
model.load_state_dict(torch.load(f'{best_epoch}.pkl'))

compute_test()
