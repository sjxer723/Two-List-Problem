from model.distribution import Distribution
from model.mallows import Mallows
import numpy as np
from utils import *

class HumanAI():
    def __init__(self, m, D_a:Distribution, D_h:Distribution):
        self.m = m
        self.D_a = D_a
        self.D_h = D_h
    
    def simulate_pick_and_choose(self, k, verbose=False):
        perm1 = self.D_a.sample()
        perm2 = self.D_h.sample()
        
        if k < 1 or k > self.m:
            raise ValueError("k must be between 1 and m")
        
        # presented k items by the algorithm
        presented_items = perm1[:k]
        indices = [perm2.index(item) for item in presented_items]
        indices.sort()
        index_of_chosen_item = indices[0]
        chosen_item = perm2[index_of_chosen_item]

        if verbose:
            print(f"Presented items: {presented_items}")
            print(f"Human's preference: {perm2}")
            print(f"Picked item {chosen_item}")
        
        return chosen_item
    
    def prob_of_picking_item(self, k, xi):
        if xi not in self.D_a.items():
            raise ValueError("unknown item")
        
        if k == 1:
            return self.D_a.prob_of_fixed_unordered_prefix([xi])
        elif k == 2:
            prob_of_picking_xi = 0
            for xj in self.D_a.items():
                if xj == xi:
                    continue
                prob_of_presenting = self.D_a.prob_of_fixed_unordered_prefix([xi, xj])
                prob_of_xi_before_xj = self.D_h.prob_of_xi_before_xj(xi, xj)
                prob_of_picking_xi += prob_of_presenting * prob_of_xi_before_xj
            return prob_of_picking_xi
        else:
            raise NotImplementedError

    def benefit_of_human_single_best(self, k, verbose=False):
        x1 = self.D_h.items()[0] # top item of human's ground truth ranking

        if k > 2:
            # Estimate the probability by sampling as the probabilty of P[xi > xj, xk] 
            # is unknown in Mallows Model
            num_of_simulation = 1000 ## TODO: get a more resonable sampling number
            prob_of_picking_x1 = 0
            for i in range(num_of_simulation):
                prob_of_picking_x1 += self.simulate_pick_and_choose(k) == x1
            prob_of_picking_x1 /= num_of_simulation
        else:
            prob_of_picking_x1 = self.prob_of_picking_item(k, x1)

        prob_of_human_picking_x1 = self.D_h.prob_of_fixed_unordered_prefix([x1])    
        if verbose:
            print("Joint system picking x1: ", prob_of_picking_x1)
            print("Human picks x1         : ", prob_of_human_picking_x1)
            print("Joint - Human          : ", prob_of_picking_x1 - prob_of_human_picking_x1)
            print("\n")

        return prob_of_picking_x1 - prob_of_human_picking_x1

    def benefit_of_human_beyond_single_best(self, m, k, utilitys, verbose=False):
        utility_of_joint_system = 0
        utility_of_human = 0
        if k > 2:
            num_of_simulation = 1000
            for i in range(num_of_simulation):
                chosen_item = self.simulate_pick_and_choose(k)
                utility_of_joint_system += utilitys[self.D_h.items().index(chosen_item)]
            
            utility_of_joint_system /= num_of_simulation
        else:
            for i in range(m):
                xi = self.D_h.items()[i] # the i-th best item of human's ground truth ranking
                prob_of_picking_xi = self.prob_of_picking_item(k, xi)
                utility_of_joint_system += prob_of_picking_xi * utilitys[i]
        
        # The utility of the human
        for i in range(m):
            xi = self.D_h.items()[i] # the i-th best item of human's ground truth ranking
            prob_of_human_picking_xi = self.D_h.prob_of_fixed_unordered_prefix([xi])
            utility_of_human += prob_of_human_picking_xi * utilitys[i]
                
        if verbose:
            print("Utility of joint system : ", utility_of_joint_system)
            print("Utility of human        : ", utility_of_human)
            print("Joint - Human          : ", utility_of_joint_system - utility_of_human)
            print("\n")

        return utility_of_joint_system - utility_of_human
    

class MultiHumanAI():
    def __init__(self, m, n, D_hs:list[Distribution], ps):
        """
            n: number of humans
            D_hs: list of distributions of humans
        """
        self.m = m
        self.n = n
        self.D_a = None
        if len(D_hs) != n:
            raise ValueError("Length of D_h must be equal to n")
        self.D_hs = D_hs
        self.ps = ps
        
    def find_layout(self):
        pi_a = []
        for D_h in self.D_hs:
            top_item = D_h.items()[0]
            if top_item not in pi_a:
                pi_a.append(top_item)
        for i in range(1, self.m + 1):
            if i not in pi_a:
                pi_a.append(i)

        self.D_a = Mallows(self.m, 1, pi_a, True)

    def benefit_of_human_single_best(self, k, verbose=False):
        benefits = []
        for i in range(self.n):
            D_h = self.D_hs[i]
            joint_system_i = HumanAI(self.m, self.D_a, D_h)
            benefits.append(joint_system_i.benefit_of_human_single_best(k, verbose))
        
        return benefits
    
    def interaction(self, N, epoch=100):
        # N is the number of interactions
        self.D_a = Mallows(self.m, 1, list(range(1, self.m+1)), True)
        credits = [0 for _ in range(self.m)]
        for t in range(N):
            coming_human_index = np.random.choice(self.n, p=self.ps)
            D_h = self.D_hs[coming_human_index]

            joint_system = HumanAI(self.m, self.D_a, D_h)
            chosen_item = joint_system.simulate_pick_and_choose(self.n)
            human_top_item = D_h.items()[0]

            if chosen_item == human_top_item:
                # postive feedback
                credits[chosen_item - 1] += 1
            else:
                # negative feedback
                credits[chosen_item - 1] -= 1

            sorted_items = sorted(range(1, self.m+1), key=lambda i: credits[i-1], reverse=True)
            self.D_a = Mallows(self.m, 1, sorted_items, True)

            if t % epoch == 0:
                benefits = self.benefit_of_human_single_best(self.n)
                info("t: {}, benefits: {}".format(t, benefits))