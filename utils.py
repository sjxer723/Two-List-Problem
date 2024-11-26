from collections import deque


def count_inversions(arr):
    inversions = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions


def count_displace(arr):
    displace = 0
    for i in range(len(arr)):
        displace += abs(i + 1 - arr[i])
    return displace

def measure(probs, upper_bound):
    res = [0  for _ in range(upper_bound)]
    for p in probs:
        # print(p, upper_bound)
        res[p] +=1
    return res 

def stirct_large(lst1, lst2, upper_bound):
    collect_1 = 0
    collect_2 = 0
    for i in range(upper_bound):
        collect_1 += lst1[i]
        collect_2 += lst2[i]
        if collect_2 > collect_1:
            return False
    return True

def transposition_distance(perm):
    n = len(perm)
    target = tuple(range(1, n+1))
    initial = tuple(perm)
    
    # 如果已经是目标排列
    if initial == target:
        return 0

    # 使用队列存储状态
    queue = deque([(initial, 0)])  # (当前排列, 已经交换的次数)
    visited = set([initial])

    while queue:
        current_perm, swaps = queue.popleft()

        # 遍历所有可能的交换
        for i in range(n):
            for j in range(i+1, n):
                # 交换i和j位置的元素
                new_perm = list(current_perm)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_perm = tuple(new_perm)

                # 如果交换后达到了目标排列
                if new_perm == target:
                    return swaps + 1

                # 如果这个排列没有被访问过，加入队列
                if new_perm not in visited:
                    visited.add(new_perm)
                    queue.append((new_perm, swaps + 1))
    
    return -1  # 如果没有找到解，返回-1


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def warn(message):
    print(bcolors.WARNING + "[WARN] " + message + bcolors.ENDC)

def header(message):
    print(bcolors.HEADER + "[HEAD] " + message + bcolors.ENDC)

def info(message):
    print(bcolors.OKBLUE + "[INFO] " + message + bcolors.ENDC)

def ok(message):
    print(bcolors.OKGREEN + "[OK] " + message + bcolors.ENDC)

def fail(message):
    print(bcolors.FAIL + "[FAIL] " + message + bcolors.ENDC)


def are_permutation_pairs_isomorphic(pair1, pair2):
    p1, p2 = pair1
    q1, q2 = pair2
    
    if len(p1) != len(q1) or len(p2) != len(q2):
        return False
    
    mapping = {}
    used_values = set()
    
    for i in range(len(p1)):
        x1, x2 = p1[i], p2[i]
        y1, y2 = q1[i], q2[i]
        
        if x1 in mapping:
            if mapping[x1] != y1:
                return False
        else:
            if y1 in used_values:
                return False
            mapping[x1] = y1
            used_values.add(y1)
        
        if x2 in mapping:
            if mapping[x2] != y2:
                return False
        else:
            if y2 in used_values:
                return False
            mapping[x2] = y2
            used_values.add(y2)
        
    return True