from utils import *
import numpy as np
import matplotlib.pyplot as plt

def init_prob_xi_before_xj(phi, N):
    """
    Initialize the probabilities that xi appears before xj.

    Parameters:
        phi: accuracy parameter, from 0+ to infty
        N: number of alternatives

    Returns:
        List: from (k = j-i+1) to the probability
    """
    exp_neg_phi = np.exp(-phi)      # exp(-phi)
    exp_neg_phi_k = np.exp(-phi)    # exp(-phi*k)
    prob_xi_before_xj_lst = []

    for k in range(2, N + 1):
        prob_xi_before_xj = -(k - 1) / (1 - exp_neg_phi_k)
        exp_neg_phi_k *= exp_neg_phi
        prob_xi_before_xj += k / (1 - exp_neg_phi_k)

        prob_xi_before_xj_lst.append(prob_xi_before_xj)

    return prob_xi_before_xj_lst

def compare_type_i_and_i_plus_one(i, N, phi, results):
    # the ranking of the two types of humans, i.e.,
    #   pi_i_least: x_i, x_1, x_2, ..., x_{i-1}, x_{i+1}, ..., x_N
    #   pi_i_1_most: x_{i+1}, x_N, ..., x_{i+2}, x_i, ..., x_1
    pi_i_least, pi_i_1_most = dict(), dict()

    pi_i_least[i] = 1
    pi_i_1_most[i + 1] = 1
    t1, t2 = 2, N

    for j in range(1, N + 1):
        if j != i:
            pi_i_least[j] = t1
            t1 += 1
        if j != i + 1:
            pi_i_1_most[j] = t2
            t2 -= 1
    prob_cmp_list = init_prob_xi_before_xj(phi, N)

    # Benefits of type i
    benefit_of_i = 0
    for j in range(1, N + 1):
        if j == i:
            continue
        prob_of_presented = np.exp(-phi * (j + i))
        gap = pi_i_least[j] - 1
        benefit_of_i += prob_of_presented * prob_cmp_list[gap - 1]

    # Benefits of type i + 1
    benefit_of_i_1 = 0
    for j in range(1, N + 1):
        if j == i + 1:
            continue
        prob_of_presented = np.exp(-phi * (j + i + 1))
        gap = pi_i_1_most[j] - 1
        benefit_of_i_1 += prob_of_presented * prob_cmp_list[gap - 1]

    # ok("Benefit of Type-{} human: {}".format(i, benefit_of_i))
    # ok("Benefit of Type-{} human: {}".format(i+1, benefit_of_i_1))
    if benefit_of_i <= benefit_of_i_1:
        # fail("Type-{} < Type-{}".format(i, i+1))
        results.append((i, phi, 'red', 'x'))
    else:
        # warn("Type-{} >= Type-{}".format(i, i+1))
        results.append((i, phi, 'green', 'o'))
    
if __name__ == '__main__':
    N = 200
    results = []
    for phi in [0.1 * i for i in range(1, 11)]:
        for i in range(1, N):
            info("i={}, phi = {}".format(i, phi))
            compare_type_i_and_i_plus_one(i, N, phi, results)
        
    # Plotting results
    plt.figure(figsize=(10, 6))
    for i, phi, color, marker in results:
        plt.scatter(i, phi, color=color, marker=marker, s=5)

    plt.xlabel("i")
    plt.ylabel("phi")
    plt.title("Comparison of Worst Type-i Human and Best Type-(i+1) Human when N = {}".format(N))
    plt.grid(True)
    plt.legend(handles=[
        plt.Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=5, label='Type-i < Type-i+1'),
        plt.Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=5, label='Type-i >= Type-i+1')
    ], loc='upper right')

    plt.show()