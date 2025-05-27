import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from gurobipy import Model, GRB, LinExpr
from model.mallows import Mallows
from utils import *
import time
import random
import json
import itertools

class StrategyMIP():
    def __init__(self, m, freqs, Ds, k, u, verbose=False):
        self.m = m                  # number of items
        self.k = k                  # number of items to be presented
        self.freqs = freqs          # frequency of each type of human
        self.Ds = Ds                # Mallows models
        self.types = len(freqs)     # number of types of human
        self.u = u
        self.W_num = m * m * m      # W(i, s, t): the probability of item i at position s after the t-th step
        self.y_num = m * m * m
        self.z_num = m * m
        self.q_num = m
        self.indicator_num = m
        self.var_num_per_human = (self.W_num + self.y_num + self.z_num + self.q_num)
        self.var_num = self.indicator_num + self.var_num_per_human * self.types
        self.model = Model()        # Create Gurobi model
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        self.used_vars = {i: False for i in range(self.var_num)}    
        self.vars = {}
        self.model.setParam('TimeLimit', 100)

    def reset_u(self, u):
        self.u = u

    def reset_freqs(self, freqs):
        self.freqs = freqs

    def create_vars(self, integer=False):
        for i in range(self.var_num):
            if integer and i <= self.m:
                self.vars[i] = self.model.addVar(vtype=GRB.INTEGER, name=f"var_{i}") if self.used_vars[i] else None
            else:
                self.vars[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"var_{i}") if self.used_vars[i] else None
        self.model.update()

    def x(self, t, i):
        central_ranking = self.Ds[t].pi_star
        return central_ranking[i - 1] - 1
    
    def W(self, _type, i, s, t):
        return self.indicator_num + _type * self.var_num_per_human + ((i - 1) * self.m * self.m) + ((s - 1) * self.m) + (t - 1)
    
    def y(self, _type, i, s, t):
        return self.indicator_num + _type * self.var_num_per_human + self.W_num + ((i - 1) * self.m * self.m) + ((s - 1) * self.m) + (t - 1)
    
    def z(self, _type, s, t):
        return self.indicator_num + _type * self.var_num_per_human + self.W_num + self.y_num + ((s - 1) * self.m) + (t - 1) 

    def q(self, _type, t):
        return self.indicator_num + _type * self.var_num_per_human + self.W_num + self.y_num + self.z_num + (t - 1)

    def add_eq(self, idxs, coeffs):
        expr = LinExpr()
        for idx, coeff in zip(idxs, coeffs):
            expr.addTerms(coeff, self.vars[idx])
        self.model.addConstr(expr == 0)

    def add_ineq(self, idxs, coeffs, lb, ub):
        expr = LinExpr()
        for idx, coeff in zip(idxs, coeffs):
            expr.addTerms(coeff, self.vars[idx])
        self.model.addConstr(expr >= lb)
        self.model.addConstr(expr <= ub)

    def add_leq(self, idxs, coeffs, b):
        expr = LinExpr()
        for idx, coeff in zip(idxs, coeffs):
            expr.addTerms(coeff, self.vars[idx])
        self.model.addConstr(expr <= b)

    def add_geq(self, idxs, coeffs, b):
        expr = LinExpr()
        for idx, coeff in zip(idxs, coeffs):
            expr.addTerms(coeff, self.vars[idx])
        self.model.addConstr(expr >= b)

    def add_binary(self, idxs):
        for idx in idxs:
            self.vars[idx].vtype = GRB.BINARY


    def record_used_vars(self):
        self.used_vars = {i: False for i in range(self.var_num)}
        for _type, _ in enumerate(self.Ds):
            for t in range(1, self.m + 1):
                for i in range(1, t):
                    for s in range(1, t+1):
                        self.used_vars[self.W(_type, i, s, t)] = True
                        self.used_vars[self.W(_type, i, s, t-1)] = True
                        self.used_vars[self.W(_type, i, s-1, t)] = True
                        self.used_vars[self.y(_type, i, s, t)] = True
                        self.used_vars[self.x(_type, i)] = True
 
                for s in range(1, t + 1):
                    # W[t, s, t] = z[s, t]
                    self.used_vars[self.W(_type, t, s, t)] = True
                    self.used_vars[self.z(_type, s, t)] = True

                    for i in range(1, t):
                        for l in range(s, self.m + 1):
                            self.used_vars[self.W(_type, i, l, t-1)] = True
                    if t == 1:
                        self.used_vars[self.z(_type, s, t)] = True
                    else:
                        self.used_vars[self.q(_type, t-1)] = True
                        self.used_vars[self.z(_type, s, t)] = True  
                        
                    # 0 <= z[s, t] <= p_{t, s} * x[t]
                    self.used_vars[self.x(_type, t)] = True
                    self.used_vars[self.z(_type, s, t)] = True

            for t in range(1, self.m):
                for i in range(1, t+1):
                    # q[t] + x[i] <= 1
                    self.used_vars[self.q(_type, t)] = True
                    self.used_vars[self.x(_type, i)] = True
                
        # 0 <= x[i] <= 1
        for i in range(1, self.indicator_num + 1):
            self.used_vars[self.x(0, i)] = True

    def solve(self, uplift=False, human_utility=0):
        """
            Solve the MIP to find the optimal strategy
        """
        for _type, D in enumerate(self.Ds):
            for t in range(1, self.m + 1):
                for i in range(1, t):
                    for s in range(1, t+1):
                        # gamma_{t, s}
                        gamma = D.Z_neg_lam(s, t) / D.Z_lam(t)
                        self.add_eq([self.W(_type, i, s, t), self.W(_type, i, s, t-1), self.y(_type, i, s, t)], [1, gamma-1, -1])
                        # gamma_{t, s-1}
                        gamma_prime = D.Z_neg_lam(s - 1, t) / D.Z_lam(t) 
                        # y[i, s, t] <= gamma_{t, s-1} * W[i, s-1, t-1]
                        self.add_leq([self.y(_type, i, s, t), self.W(_type, i, s-1, t-1)], [1, -gamma_prime], 0)
                        # y[i, s, t] <= gamma_{t, s-1} * (1 - x[i])
                        self.add_leq([self.y(_type, i, s, t), self.x(_type, i)], [1, gamma_prime], gamma_prime)

                for s in range(1, t + 1):
                    # W[t, s, t] = z[s, t]
                    self.add_eq([self.W(_type, t, s, t), self.z(_type, s, t)], [1, -1])

                    # z[s, t] <= p_{t, s}
                    w_idxs = []
                    for i in range(1, t):
                        for l in range(s, self.m + 1):
                            w_idxs.append(self.W(_type, i, l, t-1))
                    p_ts = D.Z_lst[t - s] / D.Z_lam(t)
                    if t == 1:
                        self.add_leq([self.z(_type, s, t)] + w_idxs, [1] + [-p_ts]*len(w_idxs), p_ts)
                    else:
                        self.add_leq([self.z(_type, s, t)] + w_idxs + [self.q(_type, t-1)], [1] + [-p_ts]*len(w_idxs) + [-p_ts], 0)
                        
                    # 0 <= z[s, t] <= p_{t, s} * x[t]
                    self.add_leq([self.z(_type, s, t), self.x(_type, t)], [1, -p_ts], 0)
                    self.add_geq([self.z(_type, s, t)], [1], 0)

            for t in range(1, self.m):
                for i in range(1, t+1):
                    # q[t] + x[i] <= 1
                    self.add_leq([self.q(_type, t), self.x(_type, i)], [1, 1], 1)
                    # q[t] + x[i] >= 0
                    self.add_geq([self.q(_type, t), self.x(_type, i)], [1, 1], 0)
                
                # q[t] + x[1] + x[2] + ... + x[t] >= 1
                # self.add_geq([self.q(t)] + [self.x(i) for i in range(1, t + 1)], [1] * (t + 1), 1)

        # 0 <= x[i] <= 1
        self.add_binary([self.x(0, i) for i in range(1, self.indicator_num + 1)])
        # x[1] + x[2] + ... + x[m] <= k
        self.add_leq([self.x(0, i) for i in range(1, self.indicator_num+1)], [1] * self.indicator_num, self.k)


        if uplift:
            # Add uplift constraint
            uplift_expr = LinExpr()
            for _type, D in enumerate(self.Ds):
                for i in range(1, self.m+1):
                    for s in range(1, self.m+1):
                        utility = self.u[i - 1]
                        if utility > 0:
                            uplift_expr.addTerms(utility, self.vars[self.W(_type, i, s, self.m)])
            self.model.addConstr(uplift_expr >= human_utility)

        # Set the objective
        objective_expr = LinExpr()
        for _type, D in enumerate(self.Ds):
            for i in range(1, self.m+1):
                for s in range(1, self.m+1):
                    utility = self.u[i - 1] * self.freqs[_type]
                    if utility > 0:
                        objective_expr.addTerms(utility, self.vars[self.W(_type, i, s, self.m)])
        self.model.setObjective(objective_expr, GRB.MAXIMIZE)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            solution = [v.x for v in self.model.getVars()]
            return solution[:self.m]
        else:
            fail("Failed to solve the MIP: " + str(self.model.status))
            # unfeasible
            return None

    def clear(self):
        """
            Clear the model
        """
        self.model.remove(self.model.getConstrs())
        self.model.update()

def utilities_joint_system(Ds, selected_items, u):
    """
        Calculate the social welfare of the selected items
    """
    joint_system_utilities = []
    for i, D in enumerate(Ds):
        central_ranking = D.pi_star
        utility = 0
        for g in selected_items:
            prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
            idx_of_g = central_ranking.index(g)
            utility += u[idx_of_g] * prob_selecting_g
        joint_system_utilities.append(utility)

    return joint_system_utilities


def utilities_joint_system1(Ds, D_a, k, u):
    """
        Calculate the social welfare of the selected items
    """
    joint_system_utilities = [0 for _ in range(len(Ds))]

    for selected_items in itertools.combinations(D_a.pi_star, k):
        prob_selecting = D_a.prob_of_fixed_unordered_prefix(list(selected_items))
        for i, D in enumerate(Ds):
            central_ranking = D.pi_star
            for g in selected_items:
                prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
                idx_of_g = central_ranking.index(g)
                joint_system_utilities[i] += u[idx_of_g] * prob_selecting_g * prob_selecting

    return joint_system_utilities


def utility_joint_system(D, selected_items, u):
    """
        Calculate the social welfare of the selected items
    """
    central_ranking = D.pi_star
    utility = 0
    for g in selected_items:
        prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
        idx_of_g = central_ranking.index(g)
        utility += u[idx_of_g] * prob_selecting_g

    return utility

def utilities_human(Ds, u):
    """
        Calculate the social welfare of the selected items
    """
    human_utilities = []
    for i, D in enumerate(Ds):
        central_ranking = D.pi_star
        utility = 0
        for idx, g in enumerate(central_ranking):
            prob_selecting_g = D.prob_of_xi_being_first_k(g, 1)
            utility += prob_selecting_g * u[idx]
        human_utilities.append(utility)

    return human_utilities

def test_instance1(m, _k, verbose=False, _integer=False):
    u = [1, 0.9, 0.6] + [0] * (m-3)
    pi_star1 = [1, 2, 3, 4] + [i + 5 for i in range(m-4)]
    pi_star2 = [4, 2, 3, 1] + [i + 5 for i in range(m-4)]
    Ds = [Mallows(m, 0.6, _pi_star=pi_star1), Mallows(m, 0.6, _pi_star=pi_star2)]

    start_time = time.time()
    if verbose:
        info("utilies:{}".format(u))
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
            # print(joint_system_utility, human_utility)

        if verbose:
            ok("frequency: {}, OPT: {}, Complementarity: {}".format(freqs, ", ".join(["x"+str(i) for i in selected_items]), "✅" if complementarity else "❌"))

    end_time = time.time()
    running_time = end_time - start_time

    return running_time / num_of_instances


def test_instanc2(m, _k, freq, verbose=False, shuffle=False):
    u = [1, 0.9, 0.6] + [0] * (m-3)
    pi_star1 = [1, 2, 3, 4] + [i + 5 for i in range(m-4)]
    pi_star2 = [4, 2, 3, 1] + [i + 5 for i in range(m-4)]
    Ds = [Mallows(m, 0.3, _pi_star=pi_star1), Mallows(m, 0.3, _pi_star=pi_star2)]
    StrategyMIP_instance = StrategyMIP(m, [1, 0], Ds, k=_k, u=u)
    StrategyMIP_instance.record_used_vars()
    StrategyMIP_instance.create_vars(integer=True)

    human_utilities = utilities_human(Ds, u)
    info("human_utilities:{}".format(human_utilities))

    freqs = [round(freq, 2), round(1 - freq, 2)]
    StrategyMIP_instance.reset_freqs(freqs)
    sol1 = StrategyMIP_instance.solve()
    StrategyMIP_instance.clear()
    selected_items = [i for i in range(1, m+1) if sol1[i-1] > 0]
    # Zero noise
    zero_noise_joint_system_utilies = utilities_joint_system(Ds, selected_items, u)
    print("selected_items:{}".format(selected_items))
    info("0 noise: joint_system_utilies:{}".format(zero_noise_joint_system_utilies))

    # small_utility_human_idx = np.argsort(human_utilities)[-1]
    # max_min_utility = 0
    # for phi_h in np.arange(0, 3, 0.3):
    #     T = 200
    #     joint_system_utilies_sum = [0, 0]
    #     for t in range(T):
    #         random.shuffle(selected_items)
    #         pi_star_h = selected_items + [i for i in Ds[small_utility_human_idx].pi_star if i not in selected_items]
    #         D_h = Mallows(m, phi_h, _pi_star=pi_star_h)
    #         pi_h = D_h.sample()
    #         selected_items = pi_h[:_k] 
    #         joint_system_utilies = utilities_joint_system(Ds, selected_items, u)
    #         joint_system_utilies_sum[0] += joint_system_utilies[0]
    #         joint_system_utilies_sum[1] += joint_system_utilies[1]
    #         # info("t:{}, joint_system_utilies:{}".format(t, joint_system_utilies))
    #     avg_joint_system_utilies = [x / T for x in joint_system_utilies_sum]
    #     max_min_utility = max(max_min_utility, min(avg_joint_system_utilies))
    #     info("phi_h: {}, avg_joint_system_utilies:{}".format(phi_h, avg_joint_system_utilies))

    small_utility_human_idx = 0 if zero_noise_joint_system_utilies[0] < zero_noise_joint_system_utilies[1] else 1
    # print("pi_star_a:{}".format(pi_star_a))
    print("small_utility_human_idx:{}".format(small_utility_human_idx))

    if min(zero_noise_joint_system_utilies) > min(human_utilities):
        info("0 noise: joint_system_utilies:{}".format(zero_noise_joint_system_utilies))
        return 0
    
    mini_noise = None
    phi_a = 3 
    delta = 1
    last_avg_joint_system_min = 0
    step_size = 0.1
    while mini_noise is None and phi_a > 0 and abs(delta) > 0.005:
        avg_joint_system_min = 0
        T = 10
        for _ in range(T):
            if shuffle:
                random.shuffle(selected_items)
            pi_star_a = selected_items + [i for i in Ds[small_utility_human_idx].pi_star if i not in selected_items]    
            D_a = Mallows(m, phi_a, _pi_star=pi_star_a)
            joint_system_utilies = utilities_joint_system1(Ds, D_a, _k, u)
            joint_system_min = min(joint_system_utilies)
            avg_joint_system_min += joint_system_min
        avg_joint_system_min /= T

        delta = last_avg_joint_system_min - avg_joint_system_min
        last_avg_joint_system_min = avg_joint_system_min
        info("freq: {}, phi_a: {}, joint_system_utilies:{}".format(freq, phi_a, joint_system_utilies))
        if avg_joint_system_min > min(human_utilities):
            mini_noise = np.exp(-phi_a)
            break
        phi_a -= abs(delta) * step_size

    if mini_noise is None:
        return 1
    else:
        return mini_noise
    # return min(human_utilities), min(zero_noise_joint_system_utilies), max(max_min_utility, min(zero_noise_joint_system_utilies))


def test_instance3(m, _k, verbose=False):
    u = [1, 0.9, 0.6] + [0] * (m-3)
    pi_star1 = [1, 2, 3, 4] + [i + 5 for i in range(m-4)]
    pi_star2 = [4, 2, 3, 1] + [i + 5 for i in range(m-4)]
    Ds = [Mallows(m, 0.8, _pi_star=pi_star1), Mallows(m, 0.8, _pi_star=pi_star2)]

    start_time = time.time()
    if verbose:
        info("utilies:{}".format(u))
        info("pi*(1) :{}".format(pi_star1))
        info("pi*(2) :{}".format(pi_star2))
    StrategyMIP_instance = StrategyMIP(m, [1, 0], Ds, k=_k, u=u)
    StrategyMIP_instance.record_used_vars()
    StrategyMIP_instance.create_vars()

    human_utilities = utilities_human(Ds, u)
    freq_range = np.arange(0, 1.1, 0.2)
    num_of_instances = len(freq_range)
    
    joint_system_utilitity_without_uplift_sum = 0
    joint_system_utilitity_with_uplift_sum = 0
    
    for freq in freq_range:
        freqs = [round(freq, 2), round(1 - freq, 2)]
        StrategyMIP_instance.reset_freqs(freqs)
        sol1 = StrategyMIP_instance.solve()
        StrategyMIP_instance.clear()
        # pick the largest k items
        selected_items_without_uplift = [i for i in range(1, m+1) if sol1[i-1] > 0.5]
        sol2 = StrategyMIP_instance.solve(uplift=True, human_utility=human_utilities[0])
        StrategyMIP_instance.clear()
        if sol2 is None:
            selected_items_with_uplift = None 
        else: 
            selected_items_with_uplift = [i for i in range(1, m+1) if sol2[i-1] > 0.5]
        

        complementarity = True
        joint_system_utilities = []
        joint_system_utilities_with_uplit = []
        for i, D in enumerate(Ds):
            central_ranking = D.pi_star
            joint_system_utility = 0
            for g in selected_items_without_uplift:
                prob_selecting_g = D.prob_of_xi_before_S(g, selected_items_without_uplift)
                idx_of_g = central_ranking.index(g)
                joint_system_utility += u[idx_of_g] * prob_selecting_g
            joint_system_utilities.append(joint_system_utility * freqs[i])

            joint_system_utility_with_uplit = 0
            if selected_items_with_uplift is None:
                joint_system_utilities_with_uplit.append(0)
            else:
                for g in selected_items_with_uplift:
                    prob_selecting_g = D.prob_of_xi_before_S(g, selected_items_with_uplift)
                    idx_of_g = central_ranking.index(g)
                    joint_system_utility_with_uplit += u[idx_of_g] * prob_selecting_g
                joint_system_utilities_with_uplit.append(joint_system_utility_with_uplit * freqs[i])

        if verbose:
            ok("frequency: {}, OPT: {}".format(freqs, ", ".join(["x"+str(i) for i in selected_items_without_uplift])))
            ok("frequency: {}, OPT: {}".format(freqs, ", ".join(["x"+str(i) for i in selected_items_with_uplift])))
            ok("frequency: {}, joint utilities: {}, joint utilities with uplift: {}".format(freqs, ", ".join([str(x) for x in joint_system_utilities]), ", ".join([str(x) for x in joint_system_utilities_with_uplit])))

        joint_system_utilitity_without_uplift_sum += sum(joint_system_utilities)
        joint_system_utilitity_with_uplift_sum += sum(joint_system_utilities_with_uplit)

    social_welfare_without_uplift_sum  = joint_system_utilitity_without_uplift_sum / num_of_instances
    social_welfare_with_uplift_sum = joint_system_utilitity_with_uplift_sum / num_of_instances
    social_welfare_human = sum([human_utilities[i] * freq for i, freq in enumerate(freqs)])

    return social_welfare_human, social_welfare_without_uplift_sum, social_welfare_with_uplift_sum   

def test_instance4(m, misaligned_items, k, _integer=False):
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
    human_utilities = utilities_human(Ds, u)
    human_utility = 0
    for idx, g in enumerate(human_distribution_D.pi_star):
        prob_selecting_g = human_distribution_D.prob_of_xi_being_first_k(g, 1)
        human_utility += prob_selecting_g * u[idx]
    # info("human_utilities:{}".format(human_utilities))
    # info("human_utility:{}".format(human_utility))
    StrategyMIP_instance.reset_freqs(freqs)
    sol1 = StrategyMIP_instance.solve()
    StrategyMIP_instance.clear()
    selected_items = [i for i in range(1, m+1) if sol1[i-1] > 0.5]
    end_time = time.time() 

    info("Time: {}".format(end_time - start_time))
    return end_time - start_time   


def plot_fig1(m_range=np.arange(10, 80, 10), from_stored=False, _integer=False):
    running_time_dict = dict()
    plt.figure(figsize=(8, 5))
    json_file = "figs/running_time_n_int.json" if _integer else "figs/running_time_n.json"
    fig_name = "running_time_mlp_int.pdf" if _integer else "running_time_mlp.pdf"
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


def plot_fig2():
    human_mins = []
    noiseless_mins = []
    max_min_utilities = []
    mini_noises = dict()
    fig_name = "min_noise_for_uplfit.pdf"
    plt.figure(figsize=(8, 4))
    m = 8
    k = 2

    mini_noises["no_shuffle"] = []
    mini_noises["shuffle"] = []
    for shuffle in [False, True]:
        for freq in np.arange(0, 1.1, 0.1):
            freq = round(freq, 2)
            mini_noise = test_instanc2(m, k, freq, verbose=True, shuffle=shuffle)
            print(f"freq={freq}, mini_noise={mini_noise}")

            if shuffle:
                if mini_noise >= 0.999:
                    mini_noises["shuffle"].append(np.nan)  # 不绘制连线
                else:
                    mini_noises["shuffle"].append(mini_noise)
            else:
                if mini_noise >= 0.999:
                    mini_noises["no_shuffle"].append(np.nan)
                else:
                    mini_noises["no_shuffle"].append(mini_noise)
    freqs = np.arange(0, 1.1, 0.1)
        
    plt.plot(freqs, [noise + 0.0005 for noise in mini_noises["shuffle"]], label=f"shuffle", marker='o', color='#ff7f0e')
    for i, val in enumerate(mini_noises["shuffle"]):
        if np.isnan(val):
            plt.scatter(freqs[i], 0, marker='x', color='#ff7f0e', zorder=5)
            plt.text(freqs[i], 0.002, 'infeasible', fontsize=8, ha='center', color='#ff7f0e')

    plt.plot(freqs, mini_noises["no_shuffle"], label=f"no shuffle", marker='o', color='#1f77b4')
    for i, val in enumerate(mini_noises["no_shuffle"]):
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


    # plt.plot(np.arange(0, 1.1, 0.1), human_mins, label="human", marker='*')
    # plt.plot(np.arange(0, 1.1, 0.1), noiseless_mins, label="noiseless", marker='o')
    # plt.plot(np.arange(0, 1.1, 0.1), max_min_utilities, label="noisy", marker='x')
        
    # plt.xlabel('Frequency')
    # plt.ylabel('Utility')
    # plt.legend()
    # plt.title('Utility of MIP with different frequency')
    # plt.savefig('utility_mip.pdf', dpi=600)
    # plt.show()
    # plt.close()


def plot_fig3(from_stored=False):
    social_welfares = dict()
    social_welfares["human"] = []
    social_welfares["joint_system"] = []
    social_welfares["joint_system_with_uplift"] = []

    k = 3
    plt.figure(figsize=(8, 5))
    m_range = np.arange(10, 60, 10)
    if from_stored:
        with open("figs/social_welfares.json", "r") as f:
            social_welfares = json.load(f)
    else:
        for m in m_range:
            social_welfare_human, social_welfare_joint_system, social_welfare_joint_system_with_uplift = test_instance3(m, k, verbose=True)

            social_welfares["human"].append(social_welfare_human)
            social_welfares["joint_system"].append(social_welfare_joint_system)
            social_welfares["joint_system_with_uplift"].append(social_welfare_joint_system_with_uplift)
        with open("figs/social_welfares.json", "w") as f:
            json.dump(social_welfares, f, indent=4)

    plt.plot(m_range, social_welfares["human"], label="human", marker='*')
    plt.plot(m_range, social_welfares["joint_system"], label="MIP", marker='o')
    plt.plot(m_range, social_welfares["joint_system_with_uplift"], label="MIP + Uplift", marker='x')
    plt.xlabel('Number of items (m)')
    plt.ylabel('Social Welfare')
    plt.legend()
    plt.title('Social Welfare of different m under different methods')    
    plt.savefig('social_welfare_ablation.pdf', dpi=600)
    plt.show()
    plt.close()

def plot_fig4(misaligned_items=3, phi_d=3, from_stored=False):
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
            print("utilities_human: {}".format(utilities_human_list))
            max_social_welfare = 0
            max_social_welfare_with_uplift = 0
            opt = None
            opt_with_uplift = None
            for algo_perm in itertools.permutations(range(1, m + 1)):
                selected_items = list(algo_perm)[:k]
                utilities_joint_system_list = utilities_joint_system(Ds, selected_items, u)
                social_welfare = 0 
                for D in Ds:
                    human_central_ranking = D.pi_star 
                    # print(human_central_ranking[:misaligned_items])
                    prob_human = human_distribution_D.prob(human_central_ranking[:misaligned_items])
                    human_central_ranking = list(human_central_ranking) + list(range(misaligned_items + 1, m + 1))
                    social_welfare += prob_human * utility_joint_system(D, selected_items, u)
                    # print("human_central_ranking: {}, prob_human: {}, social_welfare: {}".format(human_central_ranking, prob_human, utility_joint_system(D, selected_items, u)))
                # print("algo_perm: {}, selected_items: {}, social welfare: {}".format(algo_perm, selected_items, social_welfare))
                if opt is None or social_welfare > max_social_welfare:
                    opt = selected_items
                    max_social_welfare = social_welfare
                if min(utilities_joint_system_list) > utility_human_bar:
                    if opt_with_uplift is None or social_welfare > max_social_welfare_with_uplift:
                        opt_with_uplift = selected_items
                        max_social_welfare_with_uplift = social_welfare
            info("phi_h: {}, selected_items: {}, selected_items_with_uplift: {}".format(phi_h, opt, opt_with_uplift))
            info("social welfare: {}, social_welfare_with_uplift: {}".format(max_social_welfare, max_social_welfare_with_uplift))
            social_welfares.append(max_social_welfare)
            social_welfares_with_uplift.append(max_social_welfare_with_uplift)
        social_welfares_dict = {
            "social_welfares": social_welfares,
            "social_welfares_with_uplift": social_welfares_with_uplift
        }

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
    # plt.title('Social Welfare of different phi_h under different methods')
    plt.savefig('social_welfare_phi_h_t_{}_phi_{}.pdf'.format(misaligned_items, phi_d), dpi=600)
    plt.show()


def plot_fig5(m=20, t_range=range(1, 7), from_stored=False, _integer=False):
    m = 20
    running_time_list = dict() 
    n_list = [] 
    for n in t_range:
        n_list.append(1 if n == 1 else n_list[-1] * n)
    json_file = "figs/running_time_n_int.json" if _integer else "figs/running_time_n.json"
    fig_name = "running_time_n_int.pdf" if _integer else "running_time_n.pdf"
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
                running_time_list[str(k)].append(test_instance4(m, t, k, _integer))
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

def plot_fig6():
    m = 10
    k = 2
    misaligned_items = 3
    u = [1, 0.5, 0.2, 0] + [0] * (m-misaligned_items)
    phi_h = 0.3

    # json_file = "figs/social_welfare_t_{}_phi_{}.json".format(misaligned_items, phi_d)
    # read_success = True
    # try:
    #     if from_stored:
    #         with open(json_file, "r") as f:
    #             social_welfares_dict = json.load(f)
    # except:
    #     read_success = False
    #     warn(f"Failed to read {json_file}.")

    # if not read_success or not from_stored:
    social_welfares = []
    social_welfares_with_uplift = []

    for phi_d in np.arange(0, 3, 0.2):
        all_perms = itertools.permutations(range(1, misaligned_items + 1))
        human_distribution_D = Mallows(misaligned_items, phi_d)
        Ds = []
        for perm in all_perms:
            central_ranking = list(perm) + list(range(misaligned_items + 1, m + 1))
            D = Mallows(m, phi_h, _pi_star=central_ranking)
            Ds.append(D)
        utilities_human_list = utilities_human(Ds, u)
        utility_human_bar = utilities_human_list[0]
        max_social_welfare = 0
        max_social_welfare_with_uplift = 0
        opt = None
        opt_with_uplift = None
        utilities_joint_system_list = utilities_joint_system(Ds, list(range(1, k+1)), u)
        # print("utilities_human: {}".format(utilities_joint_system_list))
        
        # print("utilities_joint_system: {}".format(utilities_joint_system_list))
        min_utility_human_idx = np.argsort(utilities_joint_system_list)[-1]
        # print(Ds[min_utility_human_idx].pi_star)
        algo_perm = list(range(1, misaligned_items + 1)) + [i for i in Ds[min_utility_human_idx].pi_star if i not in range(1, misaligned_items + 1)]
        # print(algo_perm)
        # print("algo_perm: {}".format(algo_perm))


        print("phi_d:", phi_d)
        print(utility_human_bar)
        print(min(utilities_joint_system_list))
        min_noise = 5
        for phi_a in np.arange(2.3, 0, -0.01):
            D_a = Mallows(m, phi_a, _pi_star=algo_perm)

            tweaked_utilies_joint_system_list = utilities_joint_system1(Ds, D_a, k, u)
            min_tweaked_utility = min(tweaked_utilies_joint_system_list)
            if min_tweaked_utility > utility_human_bar:
                print(min_tweaked_utility)
                min_noise = phi_a 
                break
        print("min_noise: {}".format(min_noise))


    #     for algo_perm in itertools.permutations(range(1, m + 1)):
    #         selected_items = list(algo_perm)[:k]
    #         utilities_joint_system_list = utilities_joint_system(Ds, selected_items, u)
    #         social_welfare = 0 
    #         for D in Ds:
    #             human_central_ranking = D.pi_star 
    #             # print(human_central_ranking[:misaligned_items])
    #             prob_human = human_distribution_D.prob(human_central_ranking[:misaligned_items])
    #             human_central_ranking = list(human_central_ranking) + list(range(misaligned_items + 1, m + 1))
    #             social_welfare += prob_human * utility_joint_system(D, selected_items, u)
    #             # print("human_central_ranking: {}, prob_human: {}, social_welfare: {}".format(human_central_ranking, prob_human, utility_joint_system(D, selected_items, u)))
    #         # print("algo_perm: {}, selected_items: {}, social welfare: {}".format(algo_perm, selected_items, social_welfare))
    #         if opt is None or social_welfare > max_social_welfare:
    #             opt = selected_items
    #             max_social_welfare = social_welfare
    #         if min(utilities_joint_system_list) > utility_human_bar:
    #             if opt_with_uplift is None or social_welfare > max_social_welfare_with_uplift:
    #                 opt_with_uplift = selected_items
    #                 max_social_welfare_with_uplift = social_welfare
    #     info("phi_h: {}, selected_items: {}, selected_items_with_uplift: {}".format(phi_h, opt, opt_with_uplift))
    #     info("social welfare: {}, social_welfare_with_uplift: {}".format(max_social_welfare, max_social_welfare_with_uplift))
    #     social_welfares.append(max_social_welfare)
    #     social_welfares_with_uplift.append(max_social_welfare_with_uplift)
    # social_welfares_dict = {
    #     "social_welfares": social_welfares,
    #     "social_welfares_with_uplift": social_welfares_with_uplift
    # }

    # plt.figure(figsize=(8, 5))
    # plt.plot(np.arange(0, 3, 0.2), social_welfares_dict["social_welfares"], label="OPT", 
    #         marker='o', 
    #         linewidth=3,
    #         markersize=8,
    #         antialiased=True)
    # plt.plot(np.arange(0, 3, 0.2), social_welfares_dict["social_welfares_with_uplift"], label="OPT (Uplift)",
    #         marker='x',
    #         linewidth=3,
    #         markersize=8,
    #         antialiased=True)
    # plt.xlabel(r'$\phi_h$', fontsize=20, fontweight="bold", labelpad=2)
    # plt.ylabel('Social Welfare', fontsize=20, fontweight="bold")
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # # plt.title('Social Welfare of different phi_h under different methods')
    # plt.savefig('uplift_noise.pdf', dpi=600)
    # plt.show()


def plot_running_time():
    # vary the number of items
    plot_fig1(from_stored=True)
    # vary the number of types of humans
    plot_fig5(m=20, t_range=range(1, 7), from_stored=True)

def plot_lp_relaxation_and_opt():
    # vary the number of items
    plot_fig1(from_stored=False, _integer=True)
    # vary the number of types of humans
    # plot_fig5(m=20, t_range=range(1, 7), from_stored=False, _integer=True)

def plot_social_welfare_uplift():
    plot_fig4(misaligned_items=4, phi_d=3, from_stored=False)

def plot_uplit_noise():
    m = 6
    k = 3
    u = [1, 0.9, 0.6] + [0] * (m-3)
    pi_star1 = [1, 2, 3, 4] + [i + 5 for i in range(m-4)]
    pi_star2 = [4, 2, 3, 1] + [i + 5 for i in range(m-4)]
    Ds = [Mallows(m, 0.4, _pi_star=pi_star1), Mallows(m, 0.4, _pi_star=pi_star2)]
    StrategyMIP_instance = StrategyMIP(m, [1, 0], Ds, k=_k, u=u)
    StrategyMIP_instance.record_used_vars()
    StrategyMIP_instance.create_vars(integer=True)

    human_utilities = utilities_human(Ds, u)
    info("human_utilities:{}".format(human_utilities))

    freqs = [round(freq, 2), round(1 - freq, 2)]
    StrategyMIP_instance.reset_freqs(freqs)
    sol1 = StrategyMIP_instance.solve()
    StrategyMIP_instance.clear()
    selected_items = [i for i in range(1, m+1) if sol1[i-1] > 0.5]
    # Zero noise
    zero_noise_joint_system_utilies = utilities_joint_system(Ds, selected_items, u)
    print("selected_items:{}".format(selected_items))
    info("0 noise: joint_system_utilies:{}".format(zero_noise_joint_system_utilies))

    small_utility_human_idx = np.argsort(human_utilities)[-1]
    max_min_utility = 0
    for phi_h in np.arange(0, 3, 0.3):
        T = 200
        joint_system_utilies_sum = [0, 0]
        for t in range(T):
            random.shuffle(selected_items)
            pi_star_h = selected_items + [i for i in Ds[small_utility_human_idx].pi_star if i not in selected_items]
            D_h = Mallows(m, phi_h, _pi_star=pi_star_h)
            pi_h = D_h.sample()
            selected_items = pi_h[:_k] 
            joint_system_utilies = utilities_joint_system(Ds, selected_items, u)
            joint_system_utilies_sum[0] += joint_system_utilies[0]
            joint_system_utilies_sum[1] += joint_system_utilies[1]
            # info("t:{}, joint_system_utilies:{}".format(t, joint_system_utilies))
        avg_joint_system_utilies = [x / T for x in joint_system_utilies_sum]
        max_min_utility = max(max_min_utility, min(avg_joint_system_utilies))
        info("phi_h: {}, avg_joint_system_utilies:{}".format(phi_h, avg_joint_system_utilies))

    return min(human_utilities), min(zero_noise_joint_system_utilies), max(max_min_utility, min(zero_noise_joint_system_utilies))


if __name__ == "__main__":
    # plot_fig1(from_stored=True)
    # plot_fig3()
    # plot_fig2()
    # plot_fig4()
    # plot_fig5()
    # plot_social_welfare_uplift()

    plot_fig2()
    # plot_running_time()
    # plot_lp_relaxation_and_opt()

    # test_instanc2(15, 2, 0.5)