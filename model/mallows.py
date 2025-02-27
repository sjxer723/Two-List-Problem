import math
import unittest
import numpy as np
from model.distribution import Distribution

class Mallows(Distribution):
    def __init__(self, _m, _phi, _pi_star=None):
        self.m = _m 
        self.phi = _phi
        if _pi_star is None:
            self.pi_star = list(range(1, _m + 1))
        else:
            if len(_pi_star) != _m:
                raise ValueError("Length of pi_star must be equal to m")    
            self.pi_star = _pi_star
        self.Z_lst = [np.exp(-_phi * i) for i in range(0, _m)]
        self.Z_lam = lambda l: sum(self.Z_lst[:l])
    
    def items(self) -> list:
        return self.pi_star
    
    def sample(self) -> list:
        """
        Sample a permutation from the Mallows model.
        """
        permutation = [] 
        for i in range(1, self.m+1):
            possible_rs = np.arange(i)
            weights = np.array([1.0])
            for j in range(i-1):
                weights = np.insert(weights, 0, weights[0] * np.exp(-self.phi))
            weights /= np.sum(weights)
            
            r = np.random.choice(possible_rs, p=weights)
            permutation.insert(r, self.pi_star[i-1])
        
        return permutation

    def prob_of_fixed_unordered_prefix(self, items):
        """
            The probability of the prefix is specified:
            probability of pi[:k] = {item[0], item[1], ..., item[k-1]} (in an arbitrary order)
        """
        k = len(items)
        if k > self.m:
            raise ValueError("k must be less than or equal to m")

        sum_of_0_to_k_minus_1 = sum([i for i in range(k)])
        sum_of_indices = sum([self.pi_star.index(item) for item in items])

        prob_of_fixed_prefix = np.exp(-self.phi * (sum_of_indices - sum_of_0_to_k_minus_1))
        prob_of_fixed_prefix *= math.prod([self.Z_lam(i + 1) / self.Z_lam(self.m - i) for i in range(k)])

        return prob_of_fixed_prefix

    def prob_of_xi_before_xj(self, xi, xj):
        i = self.pi_star.index(xi)
        j = self.pi_star.index(xj)
        if i < j:
            k = j - i + 1 

            prob_rank_xi_before_xj = 1 / 2
            if self.phi > 0:
                prob_rank_xi_before_xj = k / (1 - np.exp(- self.phi * k)) - (k - 1) / (1 - np.exp(- self.phi * (k - 1)))
        else:
            k = i - j + 1

            prob_rank_xj_before_xi = 1 / 2
            if self.phi > 0:
                prob_rank_xj_before_xi = k / (1 - np.exp(- self.phi * k)) - (k - 1) / (1 - np.exp(- self.phi * (k - 1)))
            
            prob_rank_xi_before_xj = 1 - prob_rank_xj_before_xi

        return prob_rank_xi_before_xj
    

class TestMallows(unittest.TestCase):
    def test_mallows_prob_of_fixed_unordered_prefix(self):
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

if __name__ == '__main__':
    unittest.main()