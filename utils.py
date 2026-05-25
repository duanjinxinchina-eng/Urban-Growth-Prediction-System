import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import torch
import random


def encode_onehot(labels):
    """
    One-hot encoding:
    Converts labels into consecutive numerical values starting from 0.
    Returns a list where the indices correspond to the reclassified labels.
    """
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

feature_fields =  ["dem", "slope", "cityway", "motorway", "restaurant", "hospital",
          "drugstore", "hotel", "shop", "school", "park", "coach", "government", "bank",
          "plant","agri_rich", "vege_rich", "water_rich", "city_rich"]

def prepare_data(data):
    # ====================Feature and label processing===================
    ori = data["ori"]
    target = data['target']

    ori = encode_onehot(ori)
    target = encode_onehot(target)

    scaler = MinMaxScaler()
    data[feature_fields] = scaler.fit_transform(data[feature_fields])
    feature = data[feature_fields]

    # =========================torch===================
    data_tensor = torch.FloatTensor(feature.to_numpy()) # numpy → tensor
    ori_tensor = torch.LongTensor(np.where(ori)[1])
    target_tensor = torch.LongTensor(np.where(target)[1])

    return data_tensor, ori_tensor, target_tensor

def prepare_adj(loader):
    # ==============load metrix=================
    adjacency_matrix = sp.csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
    adj2 = adjacency_matrix @ adjacency_matrix # 2nd order
    adj2.setdiag(0)  # remove self-loop
    adj2.eliminate_zeros()  # save space
    adj2_coo = adj2.tocoo()

    #=========================tensor=====================
    adj_tensor = torch.FloatTensor(adj2_coo.toarray()) # np.array(adj2_coo.todense()): uint8；float32
    # print(adj_tensor.shape)

    return adj2_coo,adj_tensor

def add_richment(data, adj_coo, cell_state):

    agri_richment = []
    vege_richment = []
    water_richment = []
    city_richment = []


    neighbors_dict = {i: [] for i in range(len(data))}
    for u, v in zip(adj_coo.row, adj_coo.col):
        neighbors_dict[u].append(v)

    for i in range(len(data)):
        neighbors = neighbors_dict[i]
        total_neighbors = len(neighbors)

        agri_count = 0
        vege_count = 0
        water_count = 0
        city_count = 0

        for j in neighbors:
            land_type = cell_state[j]  # 读取邻接点的土地类型
            if land_type == 1:
                agri_count += 1
            elif land_type == 2:
                vege_count += 1
            elif land_type == 5:
                water_count += 1
            elif land_type == 8:
                city_count += 1

        if total_neighbors == 0:
            agri_richment.append(0)
            vege_richment.append(0)
            water_richment.append(0)
            city_richment.append(0)
        else:
            agri_richment.append(agri_count / total_neighbors)
            vege_richment.append(vege_count / total_neighbors)
            water_richment.append(water_count / total_neighbors)
            city_richment.append(city_count / total_neighbors)

    data['agri_rich'] = agri_richment
    data['vege_rich'] = vege_richment
    data['water_rich'] = water_richment
    data['city_rich'] = city_richment

    return data

def calu_transfer_matrix(current_land_use, next_land_use, land_use_types):
    """
    :param current_land_use: torch
    :param next_land_use: torch
    :param land_use_types:
    :return: transition_counts, transition_probabilities
    """

    current_land_use = current_land_use.numpy()
    next_land_use = next_land_use.numpy()

    # Calculate the number of transitions for each land use type from the current year to the next year.
    transition_counts = np.zeros((land_use_types, land_use_types))
    for i in range(len(current_land_use)):
        transition_counts[current_land_use[i],next_land_use[i]] += 1

    # Calculate the transition probability metrix
    transition_probabilities = np.zeros_like(transition_counts, dtype=float)
    for i in range(transition_counts.shape[0]):
        row_sum = np.sum(transition_counts[i, :])
        if row_sum != 0:
            transition_probabilities[i, :] = transition_counts[i, :] / row_sum

    return transition_counts, transition_probabilities # numpy

#=============================restrict_factor========================================
def calu_restrict_factor(current_state, N, nclass):

    C = np.zeros([N,nclass], dtype=np.float32)
    for i in range(N):
        for j in range(nclass):
            C[i,j] = 1
    for i in range(N):
        # If the previous class is water or built-up land, no conversion is performed.
        if current_state[i] == 2 or current_state[i] == 3:
                C[i, :] = 0

    return C

#===========================random_factor======================================
def calu_random_factors(N, nclass, RA_alpha=5):
    RA = np.zeros([N,nclass],dtype=np.float32)
    for i in range(N):
        gamma = random.uniform(0, 1)
        RA[i,:] = np.power(-np.log(gamma + 1e-4), RA_alpha) + 1

    return RA

# ==================precision===================================
def OA(new_landuse, labels):

    preds = new_landuse.astype(labels.dtype)
    correct = (preds == labels).astype(np.int32)
    correct_count = np.sum(correct)
    return correct_count / len(labels)

def Kappa(y_true, y_pred, nclass):

    confuse_matrix = np.zeros([nclass,nclass], dtype=np.float32)
    for i in range(len(y_true)):
        confuse_matrix[y_true[i],y_pred[i]] = confuse_matrix[y_true[i],y_pred[i]] + 1
    sum_cols = np.sum(confuse_matrix, axis=0)
    sum_rows = np.sum(confuse_matrix, axis=1)
    sum_total = sum(sum_rows)
    po = np.trace(confuse_matrix) / sum_total
    pe = sum(sum_cols * sum_rows) / sum_total ** 2
    kappa = (po-pe) / (1-pe)
    return kappa