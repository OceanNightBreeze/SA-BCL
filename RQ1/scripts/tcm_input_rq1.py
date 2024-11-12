import sys
from merge_equiv import merge_on_row, remove_from_table
from split_faults import split
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import NearestNeighbors

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
    line = f.readline()
    bugs = 0
    while (not line == '\n'):
        l = line.strip().split(' | ')
        faults = []
        if (len(l) > 1):
            if (not l[1].isdigit()):
                faults = [bugs]
            else:
                faults = []
                for b in l[1:]:
                    faults.append(int(b))
            bugs += 1
        if (method_level):
            details = l[0].split(":")
            if (len(details) != 3):
                print("ERROR: Could not do method level evaluation, exiting...")
                quit()
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
            uuts.append((l[0].split(":"), faults))
            num_locs += 1
        i += 1
        line = f.readline()
    #print(uuts, num_locs, method_map)
    return uuts, num_locs, method_map

def construct_tests(f):
    tests = []
    num_tests = 0
    line = f.readline()
    while (not line == '\n'):
        row = line.strip().split(" ")
        tests.append(row[1] == 'PASSED')
        num_tests += 1
        line = f.readline()
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

# Rq1 -> borderline and ros
def borderline_ros(tests, num_tests, locs, bin_file, method_map):
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
        # print('Only copy×1 dataset shape %s' % Counter(label))
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
    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_resample(new_feature, new_label)
    X_res = X_res.astype(bool)
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

def fill_table_ros(tests, num_tests, locs, f, method_map):
    table = []
    groups = [[i for i in range(0, locs)]]
    counts = {"p": [0] * locs, "f": [0] * locs, "tp": 0, "tf": 0, "locs": locs}
    test_map = {}

    ttable, tlabel = borderline_ros(tests, num_tests, locs, f, method_map)
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

def read_table(file_loc, split_faults, method_level=False):
    table = None
    tests = None
    num_locs = 0
    num_tests = 0
    details = None
    method_map = None
    groups = None
    file = open(file_loc)

    print(file_loc)
    
    while (True):
        line = file.readline()
        if (line == '' or not line.startswith("#")):
            break
        elif (line.startswith("#metadata")):
            while (not line == '\n'):
                line = file.readline()
        elif (line.startswith("#tests")):
            # Constructing the table
            tests, num_tests = construct_tests(file)
        elif (line.startswith("#uuts")):
            # Getting the details of the project
            details,num_locs,method_map = construct_details(file, method_level)
        elif (line.startswith("#matrix")):
            # Filling the table
            table, groups, counts, test_map = fill_table_ros(tests, num_tests,
                                                             num_locs, file, method_map)
    file.close()
    if (split_faults):
        faults,unexposed = split(find_faults(details), table, groups)
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
    print_table(table)

