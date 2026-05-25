import os.path
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from utils import prepare_adj, prepare_data, add_richment, calu_transfer_matrix, calu_restrict_factor, calu_random_factors, OA, Kappa
from model import HspGAT
np.set_printoptions(precision=4, suppress=True)

begin_year = sys.argv[1]
end_year = sys.argv[2]
years = sys.argv[3]
workspace = sys.argv[4]

begin_year = int(begin_year)
end_year = int(end_year)
years = int(years)

loader_path = os.path.join(workspace, "adjacency_matrix.npz")
loader = np.load(loader_path)
adj_coo, adj = prepare_adj(loader)

file_path = os.path.join(workspace, 'data.csv')
file = pd.read_csv(file_path)
dataset = pd.DataFrame(file)
data = dataset.copy()
ori = data['ori']
data = add_richment(data, adj_coo, ori)
data, ori, target = prepare_data(data)


N = data.shape[0]
nclass = 4
in_dim = 19
model = HspGAT(in_dim, nclass, dropout=0.6, nheads=19)
model.load_state_dict(torch.load(f'best_model.pkl', map_location=torch.device('cpu')))# .pkl train on gpu → cpu
model.eval()

trans_counts, trans_prob = calu_transfer_matrix(ori, target, nclass)
trans_counts_peryear = trans_counts // years

results = {}
def simulation(begin_class, end_class, begin_year, periods, per_year_counts, original_dataset, adjacency_coo, adjacency):

    begin_class = begin_class.numpy()

    cell_state = begin_class
    results[begin_year] = cell_state.copy()
    output_path = os.path.join(workspace,'acc.txt')
    with open(output_path, "w") as f:
        f.write("Iteration\tOA\tKappa\n")


    for year in range(periods):
        trans_matrix = per_year_counts.copy()

        # suitability probability
        data_ca = original_dataset.copy()
        data_ca = add_richment(data_ca, adjacency_coo, cell_state)
        data_ca, begin_class, end_class = prepare_data(data_ca)
        end_class = end_class.numpy()
        suitability = model(data_ca, adjacency)
        suitability = F.softmax(suitability, dim=1)
        suitability = suitability.detach().numpy()

        # Constraint at time t
        t_restrict = calu_restrict_factor(cell_state, N, nclass)

        # Random factor
        RA = calu_random_factors(N, nclass)
        # t+1
        trans_poss = suitability * t_restrict *  RA

        trans_info = []
        for i in range(N):
            max_prob = trans_poss[i].max()
            max_index = np.argmax(trans_poss[i])
            trans_info.append([i, cell_state[i], max_prob, max_index])  # [编号, 原类型, 概率, 目标类型]

        trans_info = np.array(trans_info, dtype=np.float32)

        new = cell_state.copy()  # 初始化新土地利用状态为当前状态

        for cls in range(nclass):  # cls 是目标土地类型
            # 获取所有目标转为 cls 的记录
            class_info = trans_info[trans_info[:, 3] == cls] # 分组
            # 按概率（第3列）降序排序
            sorted_class_info = class_info[class_info[:, 2].argsort()[::-1]]

            for row in sorted_class_info:
                idx = int(row[0])          # 单元格编号
                from_type = int(row[1])    # 原始土地类型
                to_type = cls              # 目标土地类型
                if trans_matrix[from_type, to_type] > 0:
                    new[idx] = to_type
                    trans_matrix[from_type, to_type] -= 1

        # 评估指标
        acc_OA = OA(new, end_class)
        kappa = Kappa(end_class, new, nclass)
        line = f"{begin_year+year+1}\t{acc_OA:.4f}\t{kappa:.4f}\n"
        print(f"OA：{acc_OA:.4f}, kappa:{kappa:.4f}")
        with open(output_path, "a") as f:
            f.write(line)
        # 更新
        cell_state = new
        results[begin_year+year+1] = new.copy()

simulation(ori, target, begin_year, years, trans_counts_peryear, dataset, adj_coo, adj)
target = target.numpy()
results[end_year+1] = target.copy() # 这是目标真实值

df = pd.DataFrame(results)
df.index.name = "id"  # 设置索引名称为id
output_path = os.path.join(workspace, "predicted.csv")
df.to_csv(output_path)