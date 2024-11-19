<<<<<<< HEAD
import sys
import re
import os
from collections import Counter

import numpy as np
from cleanlab.pruning import get_noise_indices
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import output
from merge_equiv import merge_on_row, remove_from_table
from split_faults import split

def construct_details(f, method_level):
    """
    Constructs a details object containing the information related to each
    element of the form:
    [
        (<location tuple>, [<fault_num>,...] or <fault_num> or -1),
        ...
    ]
    """
    uuts = []
    num_locs = 0 # number of reported locations (methods/lines)
    i = 0 # number of actual lines
    method_map = {}
    methods = {}
    bugs = 0
    f.readline()
    for line in f:
        # 每行按:分隔
        l = line.strip().split(':')
        # 在l[0]任意位置匹配正则表达式
        r = re.search("(.*)\$(.*)#([^:]*)", l[0])
        faults = []
        if (len(l) > 2):
            if (not l[2].isdigit()):
                faults = [bugs]
            else:
                faults = []
                for b in l[2:]:
                    faults.append(int(b))
            bugs += 1
        # 方法级的method_map是 行:出现的序号
        if (method_level):
            details = [r.group(1)+"."+r.group(2), r.group(3), l[1]]
            if ((details[0], details[1]) not in methods):
                # 添加键值对，键为每一行冒号前的，值为出现的序号（不重复）
                methods[(details[0], details[1])] = num_locs
                method_map[i] = num_locs
                uuts.append((details, faults)) # append with first line number
                num_locs += 1
            else:
                method_map[i] = methods[(details[0], details[1])]
                for fault in faults:
                    if (fault not in uuts[method_map[i]][1]):
                        uuts[method_map[i]][1].append(fault)
                #uuts[method_map[i]][1].extend(faults)
        else:
            method_map[i] = i
            uuts.append(([r.group(1)+"."+r.group(2), r.group(3), l[1]], faults))
            num_locs += 1
        i += 1
    # 方法级的num_locs是不重复的数量， statement是行数
    return uuts, num_locs, method_map

def construct_tests(tests_reader):
    tests = []
    num_tests = 0
    tests_reader.readline()
    for r in tests_reader:
        row = r.strip().split(",")
        # num_tests是测试用例总数，test是每个用例的测试结果
        tests.append(row[1] == 'PASS')
        num_tests += 1
    return tests, num_tests


def borderline(X, y, k_neighbors=1):
    X_init = X.copy()
    y_init = y.copy()

    minority_class = False
    # print("minority_class:", minority_class)
    # 找到少数类样本中的边界样本
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)
    X_min = X[y == minority_class]
    # # 用来验证
    # minority_indices = np.where(y == minority_class)[0]
    # print("minority_indices", minority_indices)


    # lst_min = []
    # for i in range(0, len(y)):
    #     if (y[i] == False):
    #         lst_min.append(i)
    # lst_min = sorted(lst_min, reverse=False)
    # lst_min_copy = lst_min[:]
    # print(lst_min == lst_min_copy)
    # print("init lst_min_copy:", lst_min_copy)
    # print("init lst_min:", lst_min)
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
        # print(true_count, false_count)
        # print("true_count:", true_count)
        # print("false_count:", false_count)
        if true_count >= false_count:
            danger.append(i)
        elif false_count > true_count:
            safe.append(i)
        elif false_count == 0:
            noise.append(i)

    print("danger num:", len(danger))
    return danger

# def copy(feature, label, multi):
#     copy_feature = feature.copy()
#     copy_label = label.copy()
#     minor_feature = feature[label == False]
#     minor_label = label[label == False]
#     for _ in range(multi):
#         copy_feature = np.concatenate((copy_feature, minor_feature), axis=0)
#         copy_label = np.concatenate((copy_label, minor_label))
#     return copy_feature, copy_label

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

        ob, label = copy_data(ob_init, label_init, 11)
        print('Only copy×11 dataset shape %s' % Counter(label))
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

    # X_res, y_res = copy_data(new_feature, new_label, 11)
    sm = SMOTE(random_state=0)
    # 假设 `new_feature` 和 `new_label` 是布尔型变量
    new_feature = new_feature.astype(np.float32)  # 或者使用 np.float64 根据需要选择精度
    # new_label = new_label.astype(np.float32)


    X_res, y_res = sm.fit_resample(new_feature, new_label)
    X_res = X_res.astype(int)

    print("After smote1 majority and danger", Counter(y_res))


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
    print("b5r1cl100")
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
    return table, groups, counts, test_map


def read_table(directory, split_faults, method_level=False):
    # Getting the details of the project
    #print("constructing details")
    details,num_locs,method_map = construct_details(open(directory+"/spectra.csv"),
            method_level)
    # Constructing the table
    #print("constructing table")
    # num_tests是测试用例总数，tests是每个用例的测试结果
    tests,num_tests = construct_tests(open(directory+"/tests.csv"))
    #print("filling table")
    table,groups,counts,test_map = fill_table_ros(tests, num_tests, num_locs,
            open(directory+"/matrix.txt"), method_map)
    if (split_faults):
        faults,unexposed = split(details["faults"], table, groups)
        for i in range(len(details)):
            if (i in unexposed):
                details[i] = (details[i][0], -1)
                print("Dropped faulty UUT:", details[i][0], "due to unexposure")
            for item in faults.items():
                if (i in item[1]):
                    details[i] = (details[i][0], item[0])
        if (len(faults) == 0):
            print("No exposable faults in", file_loc)
            quit()
    return table,counts,groups,details,test_map

if __name__ == "__main__":
    # d = sys.argv[1]
    # table,locs,tests,details = read_table(d)
    # print_table(table)

    d = "E:\\Desktop\\ISSTA_init\\Time-3-4-5-8-9-10-13-14-15-16-17-18-19-20"
    print("===========directory===========: ", d)
    table,counts,groups,details,test_map = read_table(d, False, method_level=False)
=======
import sys
import re
import os
from collections import Counter

import numpy as np
from cleanlab.pruning import get_noise_indices
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import output
from merge_equiv import merge_on_row, remove_from_table
from split_faults import split

def construct_details(f, method_level):
    """
    Constructs a details object containing the information related to each
    element of the form:
    [
        (<location tuple>, [<fault_num>,...] or <fault_num> or -1),
        ...
    ]
    """
    uuts = []
    num_locs = 0 # number of reported locations (methods/lines)
    i = 0 # number of actual lines
    method_map = {}
    methods = {}
    bugs = 0
    f.readline()
    for line in f:
        # 每行按:分隔
        l = line.strip().split(':')
        # 在l[0]任意位置匹配正则表达式
        r = re.search("(.*)\$(.*)#([^:]*)", l[0])
        faults = []
        if (len(l) > 2):
            if (not l[2].isdigit()):
                faults = [bugs]
            else:
                faults = []
                for b in l[2:]:
                    faults.append(int(b))
            bugs += 1
        # 方法级的method_map是 行:出现的序号
        if (method_level):
            details = [r.group(1)+"."+r.group(2), r.group(3), l[1]]
            if ((details[0], details[1]) not in methods):
                # 添加键值对，键为每一行冒号前的，值为出现的序号（不重复）
                methods[(details[0], details[1])] = num_locs
                method_map[i] = num_locs
                uuts.append((details, faults)) # append with first line number
                num_locs += 1
            else:
                method_map[i] = methods[(details[0], details[1])]
                for fault in faults:
                    if (fault not in uuts[method_map[i]][1]):
                        uuts[method_map[i]][1].append(fault)
                #uuts[method_map[i]][1].extend(faults)
        else:
            method_map[i] = i
            uuts.append(([r.group(1)+"."+r.group(2), r.group(3), l[1]], faults))
            num_locs += 1
        i += 1
    # 方法级的num_locs是不重复的数量， statement是行数
    return uuts, num_locs, method_map

def construct_tests(tests_reader):
    tests = []
    num_tests = 0
    tests_reader.readline()
    for r in tests_reader:
        row = r.strip().split(",")
        # num_tests是测试用例总数，test是每个用例的测试结果
        tests.append(row[1] == 'PASS')
        num_tests += 1
    return tests, num_tests


def borderline(X, y, k_neighbors=1):
    X_init = X.copy()
    y_init = y.copy()

    minority_class = False
    # print("minority_class:", minority_class)
    # 找到少数类样本中的边界样本
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)
    X_min = X[y == minority_class]
    # # 用来验证
    # minority_indices = np.where(y == minority_class)[0]
    # print("minority_indices", minority_indices)


    # lst_min = []
    # for i in range(0, len(y)):
    #     if (y[i] == False):
    #         lst_min.append(i)
    # lst_min = sorted(lst_min, reverse=False)
    # lst_min_copy = lst_min[:]
    # print(lst_min == lst_min_copy)
    # print("init lst_min_copy:", lst_min_copy)
    # print("init lst_min:", lst_min)
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
        # print(true_count, false_count)
        # print("true_count:", true_count)
        # print("false_count:", false_count)
        if true_count >= false_count:
            danger.append(i)
        elif false_count > true_count:
            safe.append(i)
        elif false_count == 0:
            noise.append(i)

    print("danger num:", len(danger))
    return danger

# def copy(feature, label, multi):
#     copy_feature = feature.copy()
#     copy_label = label.copy()
#     minor_feature = feature[label == False]
#     minor_label = label[label == False]
#     for _ in range(multi):
#         copy_feature = np.concatenate((copy_feature, minor_feature), axis=0)
#         copy_label = np.concatenate((copy_label, minor_label))
#     return copy_feature, copy_label

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

        ob, label = copy_data(ob_init, label_init, 11)
        print('Only copy×11 dataset shape %s' % Counter(label))
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

    # X_res, y_res = copy_data(new_feature, new_label, 11)
    sm = SMOTE(random_state=0)
    # 假设 `new_feature` 和 `new_label` 是布尔型变量
    new_feature = new_feature.astype(np.float32)  # 或者使用 np.float64 根据需要选择精度
    # new_label = new_label.astype(np.float32)


    X_res, y_res = sm.fit_resample(new_feature, new_label)
    X_res = X_res.astype(int)

    print("After smote1 majority and danger", Counter(y_res))


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
    print("b5r1cl100")
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
    return table, groups, counts, test_map


def read_table(directory, split_faults, method_level=False):
    # Getting the details of the project
    #print("constructing details")
    details,num_locs,method_map = construct_details(open(directory+"/spectra.csv"),
            method_level)
    # Constructing the table
    #print("constructing table")
    # num_tests是测试用例总数，tests是每个用例的测试结果
    tests,num_tests = construct_tests(open(directory+"/tests.csv"))
    #print("filling table")
    table,groups,counts,test_map = fill_table_ros(tests, num_tests, num_locs,
            open(directory+"/matrix.txt"), method_map)
    if (split_faults):
        faults,unexposed = split(details["faults"], table, groups)
        for i in range(len(details)):
            if (i in unexposed):
                details[i] = (details[i][0], -1)
                print("Dropped faulty UUT:", details[i][0], "due to unexposure")
            for item in faults.items():
                if (i in item[1]):
                    details[i] = (details[i][0], item[0])
        if (len(faults) == 0):
            print("No exposable faults in", file_loc)
            quit()
    return table,counts,groups,details,test_map

if __name__ == "__main__":
    # d = sys.argv[1]
    # table,locs,tests,details = read_table(d)
    # print_table(table)

    d = "E:\\Desktop\\ISSTA_init\\Time-3-4-5-8-9-10-13-14-15-16-17-18-19-20"
    print("===========directory===========: ", d)
    table,counts,groups,details,test_map = read_table(d, False, method_level=False)
>>>>>>> 260e7d52b (first)
