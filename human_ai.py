from distribution import Distribution
from mallows import Mallows
import unittest

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
        if k > 2:
            raise NotImplementedError

        x1 = self.D_h.items()[0] # top item of human's ground truth ranking
        prob_of_picking_x1 = self.prob_of_picking_item(k, x1)
        prob_of_human_picking_x1 = self.D_h.prob_of_fixed_unordered_prefix([x1])
    
        if verbose:
            print("Joint system picking x1: ", prob_of_picking_x1)
            print("Human picks x1         : ", prob_of_human_picking_x1)
            print("Joint - Human          : ", prob_of_picking_x1 - prob_of_human_picking_x1)
            print("\n")

        return prob_of_picking_x1 - prob_of_human_picking_x1

    def benefit_of_human_beyond_single_best(self, m, k, utilitys, verbose=False):
        if k > 2:
            raise NotImplementedError

        utility_of_joint_system = 0
        utility_of_human = 0
        
        for i in range(m):
            xi = self.D_h.items()[i] # the i-th best item of human's ground truth ranking
            prob_of_picking_xi = self.prob_of_picking_item(k, xi)
            prob_of_human_picking_xi = self.D_h.prob_of_fixed_unordered_prefix([xi])
            
            utility_of_joint_system += prob_of_picking_xi * utilitys[i]
            utility_of_human += prob_of_human_picking_xi * utilitys[i]

        if verbose:
            print("Utility of joint system : ", utility_of_joint_system)
            print("Utility of human        : ", utility_of_human)
            print("Joint - Human          : ", utility_of_joint_system - utility_of_human)
            print("\n")

        return utility_of_joint_system - utility_of_human

class TestHumanAI(unittest.TestCase):
    def test_humanai_prob_of_picking_item(self):            
        m = 5
        D_a = Mallows(m, 1)
        D_h = Mallows(m, 1)

        joint_system = HumanAI(m, D_a, D_h)
        assert(joint_system.prob_of_picking_item(1, 1) == D_a.prob_of_fixed_unordered_prefix([1]))
        assert(joint_system.prob_of_picking_item(2, 1) > D_h.prob_of_fixed_unordered_prefix([2]))

    def test_humanai_benefit_of_human_single_best(self):
        m = 10
        D_a = Mallows(m, 1)
        D_h = Mallows(m, 1)
        
        joint_system = HumanAI(m, D_a, D_h)
        benefit1 = joint_system.benefit_of_human_single_best(2)
        assert(benefit1 > 0)


if __name__ == '__main__':
    unittest.main()