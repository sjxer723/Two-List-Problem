
import math
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import * 


def init_P_sigma_i(configs, model):
    # calculate P[sigam^i]
    n = configs['n']
    P_sigma_list = [0 for _ in range(n)]

    if model == 'gRUMbel':
        try:
            v1 = configs['v1']
            beta = configs['beta']
        except:
            fail("Missing info in the config file configs/opk-{}.yaml".format(model))
            return []

        exp_v1_div_beta = math.exp(v1/beta)
        P_sigma_list[0] = exp_v1_div_beta / (exp_v1_div_beta + n - 1)
        for i in range(1, n):
            # P[σ^i] = exp(v1/β) / (exp(v1/β) + n - 1) * 
            #          Pi_{k=1}^{i-1} 1 / (1 + (exp(v1/β)−1) / (n-k))
            P_sigma_list[i] = P_sigma_list[i-1] / (1 + (exp_v1_div_beta - 1)/ (n - i))
        return P_sigma_list
    elif model == "SuperStar":
        try:
            P_sigma_list = configs['p_sigma']
            return P_sigma_list
        except:
            fail("Missing info in the config file configs/opk-{}.yaml".format(model))
    else:
        fail("Unsupported model: {}".format(model))

    return []

def init_P_k(configs, model, P_siga_list):
    n = configs['n']
    P_k = [0 for _ in range(n)]

    prefix_sum_of_P_sigma_i = 0


    for k in range(n):
        prefix_sum_of_P_sigma_i += P_siga_list[k]
        inner_weighted_sum = 0

        weight = 1
        for i in range(n - k):
            if i > 0:
                weight = weight * (n - i - k) / (n - i)
            # note: weight = math.comb(n - (i + 1), k) / math.comb(n-1, k)     
            inner_weighted_sum += weight * P_siga_list[i]

        P_k[k] = prefix_sum_of_P_sigma_i * inner_weighted_sum

    return P_k

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', default='gRUMbel')    
    
    args = parser.parse_args()

    configs = dict()
    test_n = [10 * i for i in range(1, 10)] + [100 * i for i in range(1, 10)]
    if args.model == 'gRUMbel':
        for n in test_n:
            configs['n'] = n
            v1s = [0.01 * i for i in range(1, 1000)]
            opt_ks = []
            for v1 in v1s:
                configs['v1'] = v1
                configs['beta'] = 1

                P_sigma_list = init_P_sigma_i(configs, args.model)
                if P_sigma_list == []:
                    fail("Fail to initialize P_sigma_list.")
                    return
                # print(P_sigma_list)

                P_k = init_P_k(configs, args.model, P_sigma_list)
                # print(P_k)

                opt_k = np.argmax(P_k) + 1
                # print("optimal k: ", opt_k)
                opt_ks.append(opt_k)
            
            plt.figure(figsize=(10, 6))
            plt.plot(v1s, opt_ks, label=f'n={n}', linewidth=2)
            plt.title(f'Optimal k vs v1 for n={n}', fontsize=14)
            plt.xlabel('v1/beta', fontsize=12)
            plt.ylabel('Optimal k', fontsize=12)
            plt.grid(True)
            plt.legend()
            plt.savefig("figs/%s/opt-k-n=%s.png" % (args.model, str(n).zfill(3)), dpi=600)

if __name__ == "__main__":
    _main()