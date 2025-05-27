import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import random
import time 
import matplotlib.patches as mpatches
from human_ai import *
from mip import StrategyMIP
from utils import *
from model.mallows import Mallows
from model.rum import PlacketLuce

# Figure 3
def plot_comparison_of_a1_a3():
    m = 3
    pi_a1 = [1, 2, 3] + [i for i in range(4, m + 1)]
    pi_a3 = [3, 1, 2] + [i for i in range(4, m + 1)]
    pi_h = [i + 1 for i in range(m)]
    u = [1] + [0] * (m - 1)

    phi_a_vals = np.arange(0, 5, 0.01)
    phi_h_vals = np.arange(0, 0.5, 0.01)
    result = np.zeros((len(phi_h_vals), len(phi_a_vals)))
        
    for i, phi_a in enumerate(phi_a_vals):
        for j, phi_h in enumerate(phi_h_vals):
            D_a1 = Mallows(m, phi_a, pi_a1)
            D_a3 = Mallows(m, phi_a, pi_a3)
            D_h = Mallows(m, phi_h, pi_h)

            utility_a1 = utility_joint_system1(D_h, D_a1, 2, u)
            utility_a3 = utility_joint_system1(D_h, D_a3, 2, u)

            result[j, i] = 1 if utility_a1 >= utility_a3 else 0 
    
    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap("coolwarm")
    legend_elements = [
        mpatches.Patch(color=cmap(1.0), label='A1 better'),
        mpatches.Patch(color=cmap(0.0), label='A3 better'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=True)
    plt.imshow(result, origin='lower', aspect='auto',
               extent=[phi_a_vals[0], phi_a_vals[-1], phi_h_vals[0], phi_h_vals[-1]],
               cmap=cmap, alpha=0.8, interpolation='none')
    plt.contour(phi_a_vals, phi_h_vals, result, levels=[0.5], colors='black', linewidths=3)
    plt.xlabel(r"$\phi_a$", fontsize=20, fontweight="bold")
    plt.ylabel(r"$\phi_h$", fontsize=20, fontweight="bold")
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.savefig("figs/comparison_a1_a3.pdf", dpi=600)
    plt.tight_layout()
    plt.show()

## Figure 4.1
def comparion_between_aligned_and_misaligned_algorithm():
    m = 4
    phi_a = 0.5
    phi_h = 0.5
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    idx_2 = 0
    cmap_blue = plt.get_cmap('Blues',7)
    cmap_red = plt.get_cmap('Reds',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    D_h = Mallows(m, phi_h)
    D_a_baseline = Mallows(m, phi_a)
    for algo_perm in algo_perms:
        D_a = Mallows(m, phi_a, algo_perm)
        if list(algo_perm).index(1) > 1:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utility_baseline = utility_joint_system1(D_h, D_a_baseline, 2, u)
            utilities.append(utility - utility_baseline)
        if list(algo_perm).index(1) == 0:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
            idx_1 += 1
        else:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_red(idx_2 + 1))
            idx_2 += 1

    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Aligned', fontsize=20, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_aligned_misaligned.pdf", dpi=600)
    plt.show()


# Figure 4.2
def comparion_between_aligned_algorithms():
    m = 4
    phi_a = 0.5
    phi_h = 0.5
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    cmap_blue = plt.get_cmap('Blues',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    D_h = Mallows(m, phi_h)
    D_a_baseline = Mallows(m, phi_a)
    for algo_perm in algo_perms:
        D_a = Mallows(m, phi_a, algo_perm)
        if list(algo_perm).index(1) > 0:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utility_baseline = utility_joint_system1(D_h, D_a_baseline, 2, u)
            utilities.append(utility - utility_baseline)
        plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
        idx_1 += 1

    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Aligned', fontsize=20, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_aligned_algorithms.pdf", dpi=600)
    plt.show()

# Figure 5.1 (RUM)
def comparion_between_aligned_and_misaligned_algorithm_RUM():
    m = 4
    noise_a = 0.1   # noise of the RUM model for algorithm
    noise_h = 0.1   # noise of the RUM model for human
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    idx_2 = 0
    cmap_blue = plt.get_cmap('Blues',7)
    cmap_red = plt.get_cmap('Reds',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    for algo_perm in algo_perms:
        if list(algo_perm).index(1) > 1:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            D_a_baseline = PlacketLuce(m, u, noise_a)
            D_a = PlacketLuce(m, u, noise_a, _pi_star=list(algo_perm))
            D_h = PlacketLuce(m, u, noise_h)
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utility_baseline = utility_joint_system1(D_h, D_a_baseline, 2, u)
            utilities.append(utility - utility_baseline)
        if list(algo_perm).index(1) == 0:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
            idx_1 += 1
        else:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_red(idx_2 + 1))
            idx_2 += 1

    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Aligned', fontsize=20, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_aligned_misaligned_RUM.pdf", dpi=600)
    plt.show()


# Figure 5.2 (RUM)
def comparion_between_aligned_algorithms_RUM():
    m = 4
    noise_a = 0.1   # noise of the RUM model for algorithm
    noise_h = 0.1   # noise of the RUM model for human
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    cmap_blue = plt.get_cmap('Blues',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    for algo_perm in algo_perms:
        if list(algo_perm).index(1) > 0:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            D_a_baseline = PlacketLuce(m, u, noise_a)
            D_a = PlacketLuce(m, u, noise_a, _pi_star=list(algo_perm))
            D_h = PlacketLuce(m, u, noise_h)
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utility_baseline = utility_joint_system1(D_h, D_a_baseline, 2, u)
            utilities.append(utility - utility_baseline)
        plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
        idx_1 += 1
        
    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Aligned', fontsize=20, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_aligned_algorithms_RUM.pdf", dpi=600)
    plt.show()


## Figure 6.1
def comparion_between_algorithms_over_noncollaboration():
    m = 4
    phi_a = 0.5
    phi_h = 0.5
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    idx_2 = 0
    cmap_blue = plt.get_cmap('Blues',7)
    cmap_red = plt.get_cmap('Reds',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    D_h = Mallows(m, phi_h)
    for algo_perm in algo_perms:
        D_a = Mallows(m, phi_a, algo_perm)
        if list(algo_perm).index(1) > 1:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            utility_non_collaboration = utilities_human([D_h], u)[0]
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utilities.append(utility - utility_non_collaboration)
        if list(algo_perm).index(1) == 0:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
            idx_1 += 1
        else:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_red(idx_2 + 1))
            idx_2 += 1

    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Non-collaboration', fontsize=12, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_between_algos_over_noncollaboration.pdf", dpi=600)
    plt.show()


# Figure 6.2
def comparion_between_aligned_algorithms_over_noncollaboration():
    m = 4
    phi_a = 0.5
    phi_h = 0.5
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    cmap_blue = plt.get_cmap('Blues',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    D_h = Mallows(m, phi_h)
    for algo_perm in algo_perms:
        D_a = Mallows(m, phi_a, algo_perm)
        if list(algo_perm).index(1) > 0:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            utility_non_collaboration = utilities_human([D_h], u)[0]
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utilities.append(utility - utility_non_collaboration)
        plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
        idx_1 += 1
        
    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Non-collaboration', fontsize=12, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_between_aligned_algos_over_noncollaboration.pdf", dpi=600)
    plt.show()

# Figure 7.1 (RUM)
def comparion_between_algorithms_over_noncollaboration_RUM():
    m = 4
    noise_a = 0.1   # noise of the RUM model for algorithm
    noise_h = 0.1   # noise of the RUM model for human
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    idx_2 = 0
    cmap_blue = plt.get_cmap('Blues',7)
    cmap_red = plt.get_cmap('Reds',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    for algo_perm in algo_perms:
        if list(algo_perm).index(1) > 1:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            D_a = PlacketLuce(m, u, noise_a, _pi_star=list(algo_perm))
            D_h = PlacketLuce(m, u, noise_h)
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utility_non_collaboration = utilities_human([D_h], u)[0]
            utilities.append(utility - utility_non_collaboration)
        if list(algo_perm).index(1) == 0:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
            idx_1 += 1
        else:
            plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_red(idx_2 + 1))
            idx_2 += 1

    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Non-collaboration', fontsize=12, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_between_algos_over_noncollaboration_RUM.pdf", dpi=600)
    plt.show()


# Figure 7.2 (RUM)
def comparion_between_aligned_algorithms_over_noncollaboration_RUM():
    m = 4
    noise_a = 0.1   # noise of the RUM model for algorithm
    noise_h = 0.1   # noise of the RUM model for human
    beta_range = np.arange(0.01, 11, 0.01)

    plt.figure(figsize=(8, 5))
    
    idx_1 = 0 
    cmap_blue = plt.get_cmap('Blues',7)
    algo_perms = itertools.permutations(range(1, m + 1))
    for algo_perm in algo_perms:
        if list(algo_perm).index(1) > 0:
            continue
        utilities = []
        for beta in beta_range:
            u = [np.exp(-beta * i) / np.sum(np.exp(-beta * np.arange(1, m+1))) for i in range(1, m+1)]
            D_a = PlacketLuce(m, u, noise_a, _pi_star=list(algo_perm))
            D_h = PlacketLuce(m, u, noise_h)
            utility = utility_joint_system1(D_h, D_a, 2, u)
            utility_non_collaboration = utilities_human([D_h], u)[0]
            utilities.append(utility - utility_non_collaboration)
        plt.plot(beta_range, utilities, label=" ".join([str(i) for i in algo_perm]), linewidth=2, color=cmap_blue(idx_1 + 1))
        idx_1 += 1
        
    plt.xscale('log')
    plt.xlabel(r'item value hetereogenity ($\beta$)'.format(), fontsize=20, fontweight="bold")
    plt.ylabel('Utility Change vs. Non-collaboration', fontsize=12, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig("figs/comparison_between_aligned_algos_over_noncollaboration_RUM.pdf", dpi=600)
    plt.show()


## Figure 8 and Figure 9
def plot_running_time(from_stored=False):
    """
        Plot the running time of MIP with varying number of items (denoted by m) and types of humans (denoted by n)
        Arg:
            from_stored: whether to read the running time from the stored json file
    """
    def test_instance1(m, _k, verbose=False, _integer=False):
        u = [1, 0.9, 0.6] + [0] * (m-3)
        pi_star1 = [1, 2, 3, 4] + [i + 5 for i in range(m-4)]
        pi_star2 = [4, 2, 3, 1] + [i + 5 for i in range(m-4)]
        Ds = [Mallows(m, 0.6, _pi_star=pi_star1), Mallows(m, 0.6, _pi_star=pi_star2)]

        start_time = time.time()
        if verbose:
            info("utilities:{}".format(u))
            info("pi*(1) :{}".format(pi_star1))
            info("pi*(2) :{}".format(pi_star2))
        StrategyMIP_instance = StrategyMIP(m, [1, 0], Ds, k=_k, u=u)
        StrategyMIP_instance.record_used_vars()
        StrategyMIP_instance.create_vars(_integer)

        freq_range = np.arange(0, 1.1, 0.4)
        num_of_instances = len(freq_range)
        for freq in freq_range:
            freqs = [round(freq, 2), round(1 - freq, 2)]
            StrategyMIP_instance.reset_freqs(freqs)
            sol1 = StrategyMIP_instance.solve()
            if sol1 is None:
                continue
            StrategyMIP_instance.clear()
            selected_items = [i for i in range(1, m+1) if sol1[i-1] > 0.5]
            complementarity = True
            for i, D in enumerate(Ds):
                central_ranking = D.pi_star
                joint_system_utility = 0
                for g in selected_items:
                    prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
                    idx_of_g = central_ranking.index(g)
                    joint_system_utility += u[idx_of_g] * prob_selecting_g
                
                human_utility = 0
                for idx, g in enumerate(central_ranking):
                    prob_selecting_g = D.prob_of_xi_being_first_k(g, 1)
                    human_utility += prob_selecting_g * u[idx]
                if joint_system_utility < human_utility:
                    complementarity = False
            if verbose:
                ok("frequency: {}, OPT: {}, Complementarity: {}".format(freqs, ", ".join(["x"+str(i) for i in selected_items]), "✅" if complementarity else "❌"))
        end_time = time.time()
        running_time = end_time - start_time

        return running_time / num_of_instances

    def test_instance2(m, misaligned_items, k, _integer=False):
        beta = 1 
        u = [np.exp(-beta * i) for i in range(m)]
        phi_d = 1
        phi_h = 1
        human_distribution_D = Mallows(misaligned_items, phi_d)
        all_perms = itertools.permutations(range(1, misaligned_items + 1))
        Ds = []
        freqs = []
        for perm in all_perms:
            central_ranking = list(perm) + list(range(misaligned_items + 1, m + 1))
            D = Mallows(m, phi_h, _pi_star=central_ranking)
            Ds.append(D)
            freqs.append(human_distribution_D.prob(list(perm)))

        start_time = time.time()
        StrategyMIP_instance = StrategyMIP(m, freqs, Ds, k=k, u=u)
        StrategyMIP_instance.record_used_vars()
        StrategyMIP_instance.create_vars(integer=_integer)
        human_utility = 0
        for idx, g in enumerate(human_distribution_D.pi_star):
            prob_selecting_g = human_distribution_D.prob_of_xi_being_first_k(g, 1)
            human_utility += prob_selecting_g * u[idx]
        StrategyMIP_instance.reset_freqs(freqs)
        sol1 = StrategyMIP_instance.solve()
        StrategyMIP_instance.clear()
        end_time = time.time() 

        info("Time: {}".format(end_time - start_time))
        return end_time - start_time   

    
    def plot_mlp_with_varying_m(m_range=np.arange(10, 80, 10), from_stored=False, _integer=False):
        running_time_dict = dict()
        plt.figure(figsize=(8, 5))
        json_file = "figs/running_time_int.json" if _integer else "figs/running_time.json"
        fig_name = "figs/running_time_mlp_int.pdf" if _integer else "figs/running_time_mlp.pdf"
        read_success = True
        try:
            if from_stored:
                with open(json_file, "r") as f:
                    running_time_dict = json.load(f)
        except:
            read_success = False

        if not read_success or not from_stored:
            for k in range(2, 10, 2):
                running_time_dict[k] = []
                for m in m_range:
                    running_time = test_instance1(m, k, _integer=_integer)
                    running_time_dict[k].append(running_time)
                    info(f"Running time for m={m}, k={k}: {running_time:.2f} seconds")
            with open(json_file, "w") as f:
                json.dump(running_time_dict, f, indent=4)
        for k, times in running_time_dict.items():
            plt.plot(
                m_range,
                times,
                label=f'k={k}',
                marker='o', 
                linewidth=3,
                markersize=8,
                antialiased=True
            )
        plt.ylabel('Avg. running time', fontsize=20, fontweight="bold")
        plt.xlabel('m', fontsize=20, fontweight="bold")
        plt.legend(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_name, dpi=600)
        plt.show()
        plt.close()


    def plot_mlp_with_varying_n(m=20, t_range=range(1, 7), from_stored=False, _integer=False):
        m = 20
        running_time_list = dict() 
        n_list = [] 
        for n in t_range:
            n_list.append(1 if n == 1 else n_list[-1] * n)
        json_file = "figs/running_time_n_int.json" if _integer else "figs/running_time_n.json"
        fig_name = "figs/running_time_n_int.pdf" if _integer else "figs/running_time_n.pdf"
        plt.figure(figsize=(8, 5))
        read_success = True
        try:
            if from_stored:
                with open(json_file, "r") as f:
                    running_time_list = json.load(f)
                    t_range = range(1, len(running_time_list) + 1)
        except:
            read_success = False
            warn(f"Failed to read {json_file}.")
        if not read_success or not from_stored:
            for k in range(2, 10, 2):
                running_time_list[str(k)] = []
            for t in t_range:
                for k in range(2, 10, 2):
                    running_time_list[str(k)].append(test_instance2(m, t, k, _integer))
            with open(json_file, "w") as f:
                json.dump(running_time_list, f, indent=4)
        for k in range(2, 10, 2):
            plt.plot(n_list, running_time_list[str(k)], label=f'k={k}', marker='o', linewidth=2, markersize=8, antialiased=True)
        plt.xlabel('n', fontsize=20, fontweight="bold")
        plt.ylabel('Running time', fontsize=20, fontweight="bold")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.legend(fontsize=14)
        plt.savefig(fig_name, dpi=600)
        plt.show()
        plt.close()

    # vary the number of items
    plot_mlp_with_varying_m(from_stored=from_stored, _integer=True)
    # vary the number of types of humans
    plot_mlp_with_varying_n(m=20, t_range=range(1, 7), from_stored=from_stored, _integer=True)

## Figure 10 to Figure 13
def plot_social_welfare_uplift(from_stored=False, verbose=False):
    """
    Plot the intension between the social welfare and the uplift
    Args:
        from_stored: whether to read the running time from the stored json file
        verbose: whether to print the details    
    """

    def test_instance(misaligned_items=3, phi_d=3, from_stored=False, verbose=False):
        """
        Set up a set of humans with misaligned ground truth rankings
        Args:
            misaligned_items: the number of misaligned items
            phi_d: the parameter of the human distribution
            from_stored: whether to read the running time from the stored json file
            verbose: whether to print the details
        """
        m = 6
        k = 3
        u = [1, 1, 0.5, 0.2, 0, 0]
        phi_h = 1
        json_file = "figs/social_welfare_t_{}_phi_{}.json".format(misaligned_items, phi_d)
        read_success = True
        try:
            if from_stored:
                with open(json_file, "r") as f:
                    social_welfares_dict = json.load(f)
        except:
            read_success = False
            warn(f"Failed to read {json_file}.")

        if not read_success or not from_stored:
            social_welfares = []
            social_welfares_with_uplift = []

            for phi_h in np.arange(0, 3, 0.2):
                all_perms = itertools.permutations(range(1, misaligned_items + 1))
                human_distribution_D = Mallows(misaligned_items, phi_d)
                Ds = []
                for perm in all_perms:
                    central_ranking = list(perm) + list(range(misaligned_items + 1, m + 1))
                    D = Mallows(m, phi_h, _pi_star=central_ranking)
                    Ds.append(D)
                utilities_human_list = utilities_human(Ds, u)
                utility_human_bar = utilities_human_list[0]
                if verbose:
                    info("utilities_human: {}".format(utilities_human_list))
                max_social_welfare = 0
                max_social_welfare_with_uplift = 0
                opt = None
                opt_with_uplift = None

                ## Search for an uplifting algorithm with the best social welfare
                for algo_perm in itertools.permutations(range(1, m + 1)):
                    selected_items = list(algo_perm)[:k]
                    utilities_joint_system_list = utilities_joint_system(Ds, selected_items, u)
                    social_welfare = 0 
                    for D in Ds:
                        human_central_ranking = D.pi_star 
                        prob_human = human_distribution_D.prob(human_central_ranking[:misaligned_items])
                        human_central_ranking = list(human_central_ranking) + list(range(misaligned_items + 1, m + 1))
                        social_welfare += prob_human * utility_joint_system(D, selected_items, u)
                    if opt is None or social_welfare > max_social_welfare:
                        opt = selected_items
                        max_social_welfare = social_welfare
                    if min(utilities_joint_system_list) > utility_human_bar:
                        if opt_with_uplift is None or social_welfare > max_social_welfare_with_uplift:
                            opt_with_uplift = selected_items
                            max_social_welfare_with_uplift = social_welfare
                if verbose:
                    info("phi_h: {}, selected_items: {}, selected_items_with_uplift: {}".format(phi_h, opt, opt_with_uplift))
                    info("social welfare: {}, social_welfare_with_uplift: {}".format(max_social_welfare, max_social_welfare_with_uplift))
                social_welfares.append(max_social_welfare)
                social_welfares_with_uplift.append(max_social_welfare_with_uplift)

            social_welfares_dict = {
                "social_welfares": social_welfares,
                "social_welfares_with_uplift": social_welfares_with_uplift
            }

        with open(json_file, "w") as f:
            json.dump(social_welfares_dict, f, indent=4)

        ## Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(0, 3, 0.2), social_welfares_dict["social_welfares"], label="OPT", 
                marker='o', 
                linewidth=3,
                markersize=8,
                antialiased=True)
        plt.plot(np.arange(0, 3, 0.2), social_welfares_dict["social_welfares_with_uplift"], label="OPT (Uplift)",
                marker='x',
                linewidth=3,
                markersize=8,
                antialiased=True)
        plt.xlabel(r'$\phi_h$', fontsize=20, fontweight="bold", labelpad=2)
        plt.ylabel('Social Welfare', fontsize=20, fontweight="bold")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('figs/social_welfare_phi_h_t_{}_phi_{}.pdf'.format(misaligned_items, phi_d), dpi=600)
        plt.show()

    test_instance(misaligned_items=3, phi_d=0.5, from_stored=from_stored, verbose=verbose)
    test_instance(misaligned_items=3, phi_d=3, from_stored=from_stored, verbose=verbose)
    test_instance(misaligned_items=4, phi_d=0.5, from_stored=from_stored, verbose=verbose)
    test_instance(misaligned_items=4, phi_d=3, from_stored=from_stored, verbose=verbose)


# Figure 14
def plot_minimum_noise_for_uplift(from_stored=False, verbose=False):
    def test_instance(m, _k, freq, verbose=False, shuffle=False):
        u = [1, 0.9, 0.6] + [0] * (m-3)
        pi_star1 = [1, 2, 3, 4] + [i + 5 for i in range(m-4)]
        pi_star2 = [4, 2, 3, 1] + [i + 5 for i in range(m-4)]
        Ds = [Mallows(m, 0.3, _pi_star=pi_star1), Mallows(m, 0.3, _pi_star=pi_star2)]
        StrategyMIP_instance = StrategyMIP(m, [1, 0], Ds, k=_k, u=u)
        StrategyMIP_instance.record_used_vars()
        StrategyMIP_instance.create_vars(integer=True)

        human_utilities = utilities_human(Ds, u)
        if verbose:
            info("human_utilities:{}".format(human_utilities))

        freqs = [round(freq, 2), round(1 - freq, 2)]
        StrategyMIP_instance.reset_freqs(freqs)

        ## Optimal solution for zero noise case
        sol1 = StrategyMIP_instance.solve()
        StrategyMIP_instance.clear()
        selected_items = [i for i in range(1, m+1) if sol1[i-1] > 0]
        zero_noise_joint_system_utilities = utilities_joint_system(Ds, selected_items, u)
        if verbose:
            print("selected_items:{}".format(selected_items))            
            info("0 noise: joint_system_utilities:{}".format(zero_noise_joint_system_utilities))

        # Find the ground-truth ranking for the algorithm:
        # 1. `small_utility_human_idx` as the human with the smaller utility
        # 2. then rank the remaining items according to the human with the smaller utility 
        small_utility_human_idx = 0 if zero_noise_joint_system_utilities[0] < zero_noise_joint_system_utilities[1] else 1
        if verbose:
            print("small_utility_human_idx:{}".format(small_utility_human_idx))
        
        ## no noise needed for uplift
        min_noise = None
        if min(zero_noise_joint_system_utilities) > min(human_utilities):
            if verbose:
                info("0 noise: joint_system_utilities:{}".format(zero_noise_joint_system_utilities))
            return 0 
        ## Find the minimum noise needed for uplift       
        phi_a = 3 
        delta = 1
        last_avg_joint_system_min = 0
        step_size = 0.1
        while min_noise is None and phi_a > 0 and abs(delta) > 0.005:
            avg_joint_system_min = 0
            T = 10
            for _ in range(T):
                if shuffle:
                    random.shuffle(selected_items)
                pi_star_a = selected_items + [i for i in Ds[small_utility_human_idx].pi_star if i not in selected_items]    
                D_a = Mallows(m, phi_a, _pi_star=pi_star_a)
                joint_system_utilities = utilities_joint_system1(Ds, D_a, _k, u)
                joint_system_min = min(joint_system_utilities)
                avg_joint_system_min += joint_system_min
            avg_joint_system_min /= T

            delta = last_avg_joint_system_min - avg_joint_system_min
            last_avg_joint_system_min = avg_joint_system_min
            if verbose:
                info("freq: {}, phi_a: {}, joint_system_utilities:{}".format(freq, phi_a, joint_system_utilities))
            if avg_joint_system_min > min(human_utilities):
                min_noise = np.exp(-phi_a)
                break
            phi_a -= abs(delta) * step_size

        if min_noise is None:
            return 1
        else:
            return min_noise

    min_noises = dict()
    fig_name = "figs/min_noise_for_uplfit.pdf"
    plt.figure(figsize=(8, 4))
    m = 8
    k = 2
    min_noises = dict()
    freqs = np.arange(0, 1.1, 0.1)
    if from_stored:
        with open("figs/min_noise_for_uplift.json", "r") as f:
            min_noises = json.load(f)
    else:
        min_noises["no_shuffle"] = []
        min_noises["shuffle"] = []
        for shuffle in [False, True]:
            for freq in freqs:
                freq = round(freq, 2)
                min_noise = test_instance(m, k, freq, verbose=verbose, shuffle=shuffle)
                print(f"freq={freq}, min_noise={min_noise}")

                if shuffle:
                    if min_noise >= 0.999:
                        min_noises["shuffle"].append(np.nan)
                    else:
                        min_noises["shuffle"].append(min_noise)
                else:
                    if min_noise >= 0.999:
                        min_noises["no_shuffle"].append(np.nan)
                    else:
                        min_noises["no_shuffle"].append(min_noise)
        with open("figs/min_noise_for_uplift.json", "w") as f:
            json.dump(min_noises, f, indent=4)

    ## Plotting   
    plt.plot(freqs, [noise + 0.0005 for noise in min_noises["shuffle"]], label=f"shuffle", marker='o', color='#ff7f0e')
    for i, val in enumerate(min_noises["shuffle"]):
        if np.isnan(val):
            plt.scatter(freqs[i], 0, marker='x', color='#ff7f0e', zorder=5)
            plt.text(freqs[i], 0.002, 'infeasible', fontsize=8, ha='center', color='#ff7f0e')
    plt.plot(freqs, min_noises["no_shuffle"], label=f"no shuffle", marker='o', color='#1f77b4')
    for i, val in enumerate(min_noises["no_shuffle"]):
        if np.isnan(val):
            plt.scatter(freqs[i], 0, marker='x', color='#1f77b4', zorder=5)
            plt.text(freqs[i], 0.002, 'infeasible', fontsize=8, ha='center', color='#1f77b4')
    plt.xlim(left=0)
    plt.ylabel(r'$\exp(-\phi_a)$', fontsize=20, fontweight="bold")
    plt.xlabel(r'$p_1$', fontsize=20, fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=600)
    plt.show()
    plt.close()
