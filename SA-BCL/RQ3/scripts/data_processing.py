<<<<<<< HEAD
import sys
import re
import os
from collections import Counter

import numpy as np
from cleanlab.pruning import get_noise_indices
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import output
from merge_equiv import merge_on_row, remove_from_table
from split_faults import split

def borderline(X, y, k_neighbors=5):
    X_init = X.copy()
    y_init = y.copy()

    minority_class = False
    # print("minority_class:", minority_class)
    # 找到少数类样本中的边界样本
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)
    X_min = X[y == minority_class]

    danger = [] # 少类中的边界下标
    safe = []
    noise = []
    for i, x in enumerate(X_min):
        # 找到 x 的 K 个最近邻样本
        knn = neigh.kneighbors([x], n_neighbors=k_neighbors + 1, return_distance=False)[0]
        knn = knn[1:]  # 将 x 本身排除掉
        # 找danger
        y_k = y[knn]
        # print("y_k:", y_k)
        true_count = np.sum(y_k)
        false_count = np.sum(1-y_k)
        if true_count >= false_count:
            danger.append(i)
        elif false_count > true_count:
            safe.append(i)
        elif false_count == 0:
            noise.append(i)

    print("danger num:", len(danger))
    return danger

def copy_data(feature, label, multi):
    # 复制标签为 false 的数据
    false_indices = np.where(label == False)[0]
    copy_indices = np.tile(false_indices, multi)  # 重复索引以复制数据
    copied_ob = feature[copy_indices]
    copied_label = label[copy_indices]
    # 将复制后的数据与原始数据拼接
    copied_ob = np.concatenate((feature, copied_ob), axis=0)
    copied_label = np.concatenate((label, copied_label), axis=0)
    return copied_ob, copied_label

def borderline_ros(tests, num_tests, locs, bin_file, method_map):
    table_wnoi = []
    lst = []
    false_lst = []

    for r in range(0, num_tests):
        row = [True] + [False]*locs
        # Construct the table row
        line = bin_file.readline()
        arr = line.strip().split()

        for i in range(0, len(arr)-1):
            if (arr[i] != "0"):
                i = method_map[i]
                row[i + 1] = row[i + 1] or True

        table_wnoi.append(row)

    # 得到特征和标签
    ob_init = np.array(table_wnoi)
    label_init = np.array(tests)
    print('Original dataset shape %s' % Counter(label_init))

    # 添加入borderline
    print("This strategy is borderline+ros")
    danger_lst= borderline(ob_init, label_init)
    danger_lst = sorted(danger_lst, reverse=True)
    print("danger_lst:", danger_lst)
    if (len(danger_lst) == 0):
        # print("There's no danger, so only ros1")
        print("There's no danger, so only copy×1")
        # 随机过采样
        # ros = RandomOverSampler(random_state=0, sampling_strategy=0.1)
        # ob, label = ros.fit_resample(ob_init, label_init)

        ob, label = copy_data(ob_init, label_init, 1)
        print('Only copy×1 dataset shape %s' % Counter(label))
        return ob, label

    # 少类的序号，小到大
    lst_min = []
    for i in range(0, len(label_init)):
         if (label_init[i] == False):
             lst_min.append(i)
    lst_min = sorted(lst_min, reverse=False)    # lstmin==minorityclass
    print("init lst_min:", lst_min)

    print("Ros majority and danger")
    print("init", Counter(label_init))

    for index in danger_lst:
        del lst_min[index]
    print("after lst", lst_min)
    # 多类＋danger少类
    new_feature = np.delete(ob_init, lst_min, axis=0)
    new_label = np.delete(label_init, lst_min)
    print("new", Counter(new_label))

    # ros = RandomOverSampler(random_state=0, sampling_strategy=0.1)
    # X_res, y_res = ros.fit_resample(new_feature, new_label)

    X_res, y_res = copy_data(new_feature, new_label, 0.5)
    print("After copy×0.5 majority and danger", Counter(y_res))


    # safe & noise 加在尾部
    sn_feature = ob_init[lst_min]
    sn_label = label_init[lst_min]
    if (len(sn_feature) == 0):
        print("There's no sn")
        return X_res, y_res
    print("num of safe and noise", len(sn_feature))
    X_res = np.concatenate((X_res, sn_feature), axis=0)
    y_res_list = list(y_res)
    sn_label_list = list(sn_label)
    y_res_list.extend(sn_label_list)
    y_res = np.array(y_res_list)

    print("After add safe and noise", Counter(y_res))

    return X_res, y_res


def cl(tests, num_tests, locs, f, method_map):
    table_wnoi = []
    lst = []
    false_lst = []
    # print("b5r1cl100")
    ob, label = borderline_ros(tests, num_tests, locs, f, method_map)

    false_count = np.sum(label == False)
    if (false_count < 10):
        print("This file has not enough failed labels (the nummbers of failed label < 10)!")
        return ob, label

    # 交叉验证
    psx = np.zeros((len(label), 2))

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(ob, label):

        # 根据索引划分训练集和测试集
        X_train, X_test = ob[train_index], ob[test_index]
        Y_train, Y_test = label[train_index], label[test_index]

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, Y_train)

        # 预测
        psx_cv = model.predict_proba(X_test)
        psx[test_index] = psx_cv

    import warnings
    warnings.filterwarnings("ignore")

    # 置信学习
    ordered = get_noise_indices(
        s=label,
        psx=psx,
        prune_method='prune_by_class',
        sorted_index_method='normalized_margin'
    )

    if (len(ordered) == 0):
        print("There's no noise!")
        return ob, label
    print("The ordered noise indices are: ", ordered)
    print("The ordered noise num is: ", len(ordered))

    # 不删标签为false的
    for i in range(0, len(ordered)):
        if (label[ordered[i]] == False):
            lst.append(i)
    print("in ordered but label false lst:", lst)
    print("lst num: ", len(lst))
    ordered = np.delete(ordered, lst)

    if (len(ordered) == 0):
        print("There's all false noise, cl nothing!")
        return ob, label

    ordered = np.sort(ordered)
    ordered_lst = ordered.tolist()
    charas = np.delete(ob, ordered_lst, axis=0)
    labels = np.delete(label, ordered_lst)

    return charas, labels

def fill_table_ros(tests, num_tests, locs, f, method_map):
    table = []
    groups = [[i for i in range(0, locs)]]
    counts = {"p": [0] * locs, "f": [0] * locs, "tp": 0, "tf": 0, "locs": locs}
    test_map = {}

    ttable, tlabel = cl(tests, num_tests, locs, f, method_map)
    for r in range(0, ttable.shape[0]):
        seen = []
        for i in range(1, (locs+1)):
            if (ttable[r][i]):
                pos = i - 1
                pos = method_map[pos]
                if (pos not in seen):
                    seen.append(pos)
                    if (tlabel[r]):
                        counts["p"][pos] += 1
                    else:
                        counts["f"][pos] += 1
        # Use row to merge equivalences
        groups = merge_on_row(ttable[r], groups)
        # Increment total counts, and append row to table
        if (tlabel[r]):
            counts["tp"] += 1
        else:
            counts["tf"] += 1
            table.append(ttable[r].tolist())
            test_map[r] = len(table) - 1
    groups.sort(key=lambda group: group[0])
    # Remove groupings from table
    remove_from_table(groups, table, counts)
=======
import sys
import re
import os
from collections import Counter

import numpy as np
from cleanlab.pruning import get_noise_indices
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import output
from merge_equiv import merge_on_row, remove_from_table
from split_faults import split

def borderline(X, y, k_neighbors=5):
    X_init = X.copy()
    y_init = y.copy()

    minority_class = False
    # print("minority_class:", minority_class)
    # 找到少数类样本中的边界样本
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)
    X_min = X[y == minority_class]

    danger = [] # 少类中的边界下标
    safe = []
    noise = []
    for i, x in enumerate(X_min):
        # 找到 x 的 K 个最近邻样本
        knn = neigh.kneighbors([x], n_neighbors=k_neighbors + 1, return_distance=False)[0]
        knn = knn[1:]  # 将 x 本身排除掉
        # 找danger
        y_k = y[knn]
        # print("y_k:", y_k)
        true_count = np.sum(y_k)
        false_count = np.sum(1-y_k)
        if true_count >= false_count:
            danger.append(i)
        elif false_count > true_count:
            safe.append(i)
        elif false_count == 0:
            noise.append(i)

    print("danger num:", len(danger))
    return danger

def copy_data(feature, label, multi):
    # 复制标签为 false 的数据
    false_indices = np.where(label == False)[0]
    copy_indices = np.tile(false_indices, multi)  # 重复索引以复制数据
    copied_ob = feature[copy_indices]
    copied_label = label[copy_indices]
    # 将复制后的数据与原始数据拼接
    copied_ob = np.concatenate((feature, copied_ob), axis=0)
    copied_label = np.concatenate((label, copied_label), axis=0)
    return copied_ob, copied_label

def borderline_ros(tests, num_tests, locs, bin_file, method_map):
    table_wnoi = []
    lst = []
    false_lst = []

    for r in range(0, num_tests):
        row = [True] + [False]*locs
        # Construct the table row
        line = bin_file.readline()
        arr = line.strip().split()

        for i in range(0, len(arr)-1):
            if (arr[i] != "0"):
                i = method_map[i]
                row[i + 1] = row[i + 1] or True

        table_wnoi.append(row)

    # 得到特征和标签
    ob_init = np.array(table_wnoi)
    label_init = np.array(tests)
    print('Original dataset shape %s' % Counter(label_init))

    # 添加入borderline
    print("This strategy is borderline+ros")
    danger_lst= borderline(ob_init, label_init)
    danger_lst = sorted(danger_lst, reverse=True)
    print("danger_lst:", danger_lst)
    if (len(danger_lst) == 0):
        # print("There's no danger, so only ros1")
        print("There's no danger, so only copy×1")
        # 随机过采样
        # ros = RandomOverSampler(random_state=0, sampling_strategy=0.1)
        # ob, label = ros.fit_resample(ob_init, label_init)

        ob, label = copy_data(ob_init, label_init, 1)
        print('Only copy×1 dataset shape %s' % Counter(label))
        return ob, label

    # 少类的序号，小到大
    lst_min = []
    for i in range(0, len(label_init)):
         if (label_init[i] == False):
             lst_min.append(i)
    lst_min = sorted(lst_min, reverse=False)    # lstmin==minorityclass
    print("init lst_min:", lst_min)

    print("Ros majority and danger")
    print("init", Counter(label_init))

    for index in danger_lst:
        del lst_min[index]
    print("after lst", lst_min)
    # 多类＋danger少类
    new_feature = np.delete(ob_init, lst_min, axis=0)
    new_label = np.delete(label_init, lst_min)
    print("new", Counter(new_label))

    # ros = RandomOverSampler(random_state=0, sampling_strategy=0.1)
    # X_res, y_res = ros.fit_resample(new_feature, new_label)

    X_res, y_res = copy_data(new_feature, new_label, 0.5)
    print("After copy×0.5 majority and danger", Counter(y_res))


    # safe & noise 加在尾部
    sn_feature = ob_init[lst_min]
    sn_label = label_init[lst_min]
    if (len(sn_feature) == 0):
        print("There's no sn")
        return X_res, y_res
    print("num of safe and noise", len(sn_feature))
    X_res = np.concatenate((X_res, sn_feature), axis=0)
    y_res_list = list(y_res)
    sn_label_list = list(sn_label)
    y_res_list.extend(sn_label_list)
    y_res = np.array(y_res_list)

    print("After add safe and noise", Counter(y_res))

    return X_res, y_res


def cl(tests, num_tests, locs, f, method_map):
    table_wnoi = []
    lst = []
    false_lst = []
    # print("b5r1cl100")
    ob, label = borderline_ros(tests, num_tests, locs, f, method_map)

    false_count = np.sum(label == False)
    if (false_count < 10):
        print("This file has not enough failed labels (the nummbers of failed label < 10)!")
        return ob, label

    # 交叉验证
    psx = np.zeros((len(label), 2))

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(ob, label):

        # 根据索引划分训练集和测试集
        X_train, X_test = ob[train_index], ob[test_index]
        Y_train, Y_test = label[train_index], label[test_index]

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, Y_train)

        # 预测
        psx_cv = model.predict_proba(X_test)
        psx[test_index] = psx_cv

    import warnings
    warnings.filterwarnings("ignore")

    # 置信学习
    ordered = get_noise_indices(
        s=label,
        psx=psx,
        prune_method='prune_by_class',
        sorted_index_method='normalized_margin'
    )

    if (len(ordered) == 0):
        print("There's no noise!")
        return ob, label
    print("The ordered noise indices are: ", ordered)
    print("The ordered noise num is: ", len(ordered))

    # 不删标签为false的
    for i in range(0, len(ordered)):
        if (label[ordered[i]] == False):
            lst.append(i)
    print("in ordered but label false lst:", lst)
    print("lst num: ", len(lst))
    ordered = np.delete(ordered, lst)

    if (len(ordered) == 0):
        print("There's all false noise, cl nothing!")
        return ob, label

    ordered = np.sort(ordered)
    ordered_lst = ordered.tolist()
    charas = np.delete(ob, ordered_lst, axis=0)
    labels = np.delete(label, ordered_lst)

    return charas, labels

def fill_table_ros(tests, num_tests, locs, f, method_map):
    table = []
    groups = [[i for i in range(0, locs)]]
    counts = {"p": [0] * locs, "f": [0] * locs, "tp": 0, "tf": 0, "locs": locs}
    test_map = {}

    ttable, tlabel = cl(tests, num_tests, locs, f, method_map)
    for r in range(0, ttable.shape[0]):
        seen = []
        for i in range(1, (locs+1)):
            if (ttable[r][i]):
                pos = i - 1
                pos = method_map[pos]
                if (pos not in seen):
                    seen.append(pos)
                    if (tlabel[r]):
                        counts["p"][pos] += 1
                    else:
                        counts["f"][pos] += 1
        # Use row to merge equivalences
        groups = merge_on_row(ttable[r], groups)
        # Increment total counts, and append row to table
        if (tlabel[r]):
            counts["tp"] += 1
        else:
            counts["tf"] += 1
            table.append(ttable[r].tolist())
            test_map[r] = len(table) - 1
    groups.sort(key=lambda group: group[0])
    # Remove groupings from table
    remove_from_table(groups, table, counts)
>>>>>>> 260e7d52b (first)
    return table, groups, counts, test_map