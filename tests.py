import unittest
from model.mallows import Mallows
from model.rum import SuperStar, gRUMBel
from human_ai import HumanAI
from utils import *

class TestMallows(unittest.TestCase):
    def test_mallows_prob_of_fixed_unordered_prefix(self):
        info("test_mallows_prob_of_fixed_unordered_prefix")
        m = 5
        pi_star = [1, 2, 3, 4, 5]
        
        phi = 0
        D = Mallows(m, phi, pi_star)
        assert(D.prob_of_fixed_unordered_prefix([1]) == 0.2)
        assert(D.prob_of_fixed_unordered_prefix([1, 2]) == 0.1)
        assert(D.prob_of_fixed_unordered_prefix([2, 3]) == 0.1)
        assert(D.prob_of_fixed_unordered_prefix([1, 2, 3]) == 0.1)
        
        phi = 10
        D = Mallows(m, phi, pi_star)
        assert(D.prob_of_fixed_unordered_prefix([1]) > 0.9)
        assert(D.prob_of_fixed_unordered_prefix([1, 2]) > 0.9)
        assert(D.prob_of_fixed_unordered_prefix([1, 2, 3]) > 0.9)
        
    def test_mallows_sample(self):
        info("test_mallows_sample")
        m = 10
        theta = 1.0 

        # no ground truth, [1,2,...,n] in default
        D = Mallows(m, theta)
        sample_perm = D.sample()
        print("Sampled permutation:", sample_perm)

        # with ground truth
        pi_star = list(range(m, 0, -1))
        D = Mallows(m, theta, pi_star)
        sample_perm2 = D.sample()
        print("Sampled permutation with pi_star =", pi_star,":", sample_perm2)

    def test_mallows_prob_of_xi_being_first_k(self):
        info("test_mallows_xi_being_first_k")
        m = 2
        theta = 0
        D = Mallows(m, theta)
        assert(D.prob_of_xi_being_first_k(1, 1) == 0.5)
        assert(D.prob_of_xi_being_first_k(1, 2) == 1)
        assert(D.prob_of_xi_being_first_k(2, 1) == 0.5)
        assert(D.prob_of_xi_being_first_k(2, 2) == 1)
        
        m = 3
        theta = 1
        D = Mallows(m, theta)
        assert(abs(D.prob_of_xi_being_first_k(1, 3) - 1) < 1e-5)
        assert(abs(D.prob_of_xi_being_first_k(2, 3) - 1) < 1e-5)
        assert(abs(D.prob_of_xi_being_first_k(3, 3) - 1) < 1e-5)

        m = 100
        theta = 1
        D = Mallows(m, theta)
        # print(D.prob_of_xi_being_first_k(10, 10))
        # print(D.prob_of_xi_being_first_k(10, 11))
        # print(D.prob_of_xi_being_first_k(10, 100))
        # print(D.prob_of_xi_being_first_k(2, 1))
        # print(D.prob_of_xi_being_first_k(2, 2))
        # print(D.prob_of_xi_being_first_k(2, 3))

        
class TestSuperStar(unittest.TestCase):
    def test_superstar_sample(self):
        info("test_superstar_sample")
        D_super = SuperStar(5, 1, [0.5, 0.4, 0.1, 0, 0])
        for _ in range(5):
            sampled_perm = D_super.sample()
            print(sampled_perm)

    def test_prob_fixed_unordered_prefix(self):
        info("test_prob_fixed_unordered_prefix")
        D_super = SuperStar(5, 1, [0.5, 0.4, 0.1, 0, 0])
        
        assert(D_super.prob_of_fixed_unordered_prefix([1, 2]) == 0.225)
        assert(D_super.prob_of_fixed_unordered_prefix([2, 3]) == 0.1 / 6)
        

    def test_prob_xi_before_xj(self):
        info("test_prob_xi_before_xj")
        D_super = SuperStar(5, 1, [0.5, 0.4, 0.1, 0, 0])
        
        assert(abs(D_super.prob_of_xi_before_xj(1, 2) - 0.85) < 1e-5)
        assert(abs(D_super.prob_of_xi_before_xj(2, 1) - 0.15) < 1e-5)
        assert(D_super.prob_of_xi_before_xj(2, 3) == 0.5)


class TestHumanAI(unittest.TestCase):
    def test_humanai_prob_of_picking_item(self):      
        info("test_humanai_prob_of_picking_item")      
        m = 5
        D_a = Mallows(m, 1)
        D_h = Mallows(m, 1)

        joint_system = HumanAI(m, D_a, D_h)
        assert(joint_system.prob_of_picking_item(1, 1) == D_a.prob_of_fixed_unordered_prefix([1]))
        assert(joint_system.prob_of_picking_item(2, 1) > D_h.prob_of_fixed_unordered_prefix([2]))

    def test_humanai_benefit_of_human_single_best(self):
        info("test_humanai_benefit_of_human_single_best")
        m = 10
        D_a = Mallows(m, 1)
        D_h = Mallows(m, 1)
        
        joint_system = HumanAI(m, D_a, D_h)
        benefit1 = joint_system.benefit_of_human_single_best(2)
        assert(benefit1 > 0)

class TestgRUMbel(unittest.TestCase):
    def test_grubel_init(self):
        info("test_grubel_init")
        g = gRUMBel(5, 1, 0, 1)
        assert(g.p_sigmas[0] == 0.2)

        g = gRUMBel(10, 1, 0, 1)
        assert(g.p_sigmas[0] == 0.1)

    def test_grubel_sample(self):
        info("test_grubel_sample")
        g = gRUMBel(5, 1, 10, 1)
        for _ in range(5):
            sampled_perm = g.sample()
            print(sampled_perm)

if __name__ == "__main__":
    # unittest.main()
    m = 5
    phi=10
    D_a = Mallows(m, phi, [1, 2, 3, 4, 5])
    D_h = Mallows(m, phi, [2, 1, 3, 4, 5])
    joint_system = HumanAI(m, D_a, D_h)
    print(joint_system.benefit_of_human_single_best(2))
