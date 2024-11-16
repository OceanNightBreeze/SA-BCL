import sys
import re
import numpy as np
from cleanlab.pruning import get_noise_indices
from imblearn.under_sampling import OneSidedSelection
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
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
        l = line.strip().split(':')
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
        if (method_level):
            details = [r.group(1)+"."+r.group(2), r.group(3), l[1]]
            if ((details[0], details[1]) not in methods):
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
    return uuts, num_locs, method_map

def construct_tests(tests_reader):
    tests = []
    num_tests = 0
    tests_reader.readline()
    for r in tests_reader:
        row = r.strip().split(",")
        tests.append(row[1] == 'PASS')
        num_tests += 1
    return tests, num_tests

def borderline(X, y, k_neighbors=5):
    minority_class = False
    # Find borderline samples of the minority class ones
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)
    X_min = X[y == minority_class]

    danger = []
    # safe = []
    # noise = []
    for i, x in enumerate(X_min):
        # K nearest neighbor
        knn = neigh.kneighbors([x], n_neighbors=k_neighbors + 1, return_distance=False)[0]
        knn = knn[1:]
        # Find danger class samples
        y_k = y[knn]
        true_count = np.sum(y_k)
        false_count = np.sum(1-y_k)
        if true_count >= false_count:
            danger.append(i)
        # elif false_count > true_count:
        #     safe.append(i)
        # elif false_count == 0:
        #     noise.append(i)
    return danger
def aug_data(feature, label, multi):
    # Copy data with label false
    false_indices = np.where(label == False)[0]
    copy_indices = np.tile(false_indices, multi)
    copied_ob = feature[copy_indices]
    copied_label = label[copy_indices]
    # Concatenate
    copied_ob = np.concatenate((feature, copied_ob), axis=0)
    copied_label = np.concatenate((label, copied_label), axis=0)
    return copied_ob, copied_label
def borderline_aug(tests, num_tests, locs, bin_file, method_map):
    table_wnoi = []

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
    # Characters and labels
    ob_init = np.array(table_wnoi)
    label_init = np.array(tests)

    # Add borderline samples
    danger_lst= borderline(ob_init, label_init, 3)
    danger_lst = sorted(danger_lst, reverse=True)
    if (len(danger_lst) == 0):
        # ob, label = copy_data(ob_init, label_init, 1)
        # return ob, label
        return ob_init, label_init

    lst_min = []
    for i in range(0, len(label_init)):
         if (label_init[i] == False):
             lst_min.append(i)
    lst_min = sorted(lst_min, reverse=False)

    for index in danger_lst:
        del lst_min[index]
    new_feature = np.delete(ob_init, lst_min, axis=0)
    new_label = np.delete(label_init, lst_min)
    X_res, y_res = aug_data(new_feature, new_label, 1)
    # Add samples of the safe class and noise class
    sn_feature = ob_init[lst_min]
    sn_label = label_init[lst_min]
    if (len(sn_feature) == 0):
        return X_res, y_res
    X_res = np.concatenate((X_res, sn_feature), axis=0)
    y_res_list = list(y_res)
    sn_label_list = list(sn_label)
    y_res_list.extend(sn_label_list)
    y_res = np.array(y_res_list)
    return X_res, y_res

def cl(tests, num_tests, locs, f, method_map):
    lst = []
    ob, label = borderline_aug(tests, num_tests, locs, f, method_map)

    false_count = np.sum(label == False)
    true_count = np.sum(label == True)
    if (false_count < 10 or true_count < 10):
        return ob, label

    # Cross validation
    psx = np.zeros((len(label), 2))
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(ob, label):
        X_train, X_test = ob[train_index], ob[test_index]
        Y_train, Y_test = label[train_index], label[test_index]
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, Y_train)
        psx_cv = model.predict_proba(X_test)
        psx[test_index] = psx_cv
    import warnings
    warnings.filterwarnings("ignore")

    # Confidence learning
    ordered = get_noise_indices(
        s=label,
        psx=psx,
        prune_method='prune_by_class',
        sorted_index_method='normalized_margin'
    )
    if (len(ordered) == 0):
        return ob, label
    # Do not delete samples with false labels
    for i in range(0, len(ordered)):
        if (label[ordered[i]] == False):
            lst.append(i)
    ordered = np.delete(ordered, lst)
    if (len(ordered) == 0):
        return ob, label
    ordered = np.sort(ordered)
    ordered_lst = ordered.tolist()
    charas = np.delete(ob, ordered_lst, axis=0)
    labels = np.delete(label, ordered_lst)
    return charas, labels

def isof(tests, num_tests, locs, f, method_map):
    lst = []
    ob, label = borderline_aug(tests, num_tests, locs, f, method_map)
    isof = IsolationForest(random_state=0).fit(ob)
    x_array = isof.predict(ob)
    count = 0
    ordered = []
    for j in range(len(x_array)):
        if x_array[j] == -1:
            ordered.append(j)
            count += 1
    if (len(ordered) == 0):
        return ob, label
    # Don't delete samples with false labels
    for i in range(0, len(ordered)):
        if (label[ordered[i]] == False):
            lst.append(i)
    ordered = np.delete(ordered, lst)
    if (len(ordered) == 0):
        return ob, label
    ordered = np.sort(ordered)
    ordered_lst = ordered.tolist()
    charas = np.delete(ob, ordered_lst, axis=0)
    labels = np.delete(label, ordered_lst)
    return charas, labels

def oss(tests, num_tests, locs, f, method_map):
    ob, label = borderline_aug(tests, num_tests, locs, f, method_map)
    oss = OneSidedSelection()
    charas, labels = oss.fit_resample(ob, label)
    return charas, labels


def fill_table(tests, num_tests, locs, f, method_map):
    table = []
    groups = [[i for i in range(0, locs)]]
    counts = {"p": [0] * locs, "f": [0] * locs, "tp": 0, "tf": 0, "locs": locs}
    test_map = {}

    # Change strategies by modifying the function name
    # (cl, isof, oss)
    ttable, tlabel = isof(tests, num_tests, locs, f, method_map)
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
    details,num_locs,method_map = construct_details(open(directory+"/spectra.csv"),
            method_level)
    # Constructing the table
    tests,num_tests = construct_tests(open(directory+"/tests.csv"))
    table,groups,counts,test_map = fill_table(tests, num_tests, num_locs,
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
    d = sys.argv[1]
    table,locs,tests,details = read_table(d)

