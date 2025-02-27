import math
import unittest
import numpy as np
from model.distribution import Distribution

class Mallows(Distribution):
    def __init__(self, _m, _phi, _pi_star=None, _fixed=False):
        self.m = _m 
        self.phi = _phi
        if _pi_star is None:
            self.pi_star = list(range(1, _m + 1))
        else:
            if len(_pi_star) != _m:
                print("pi_star: ", _pi_star)
                print("m: ", _m)
                raise ValueError("Length of pi_star must be equal to m")    
            self.pi_star = _pi_star
        self.Z_lst = [np.exp(-_phi * i) for i in range(0, _m)]
        self.Z_lam = lambda l: sum(self.Z_lst[:l])
        self.fixed = _fixed
    
    def items(self) -> list:
        return self.pi_star
    
    def sample(self) -> list:
        """
        Sample a permutation from the Mallows model.
        """
        if self.fixed == True:
            return self.pi_star

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

        if self.fixed == True:
            if sorted(items) == sorted(self.items()[:k]):
                return 1
            return 0
        
        sum_of_0_to_k_minus_1 = sum([i for i in range(k)])
        sum_of_indices = sum([self.pi_star.index(item) for item in items])

        prob_of_fixed_prefix = np.exp(-self.phi * (sum_of_indices - sum_of_0_to_k_minus_1))
        prob_of_fixed_prefix *= math.prod([self.Z_lam(i + 1) / self.Z_lam(self.m - i) for i in range(k)])

        return prob_of_fixed_prefix

    def prob_of_xi_before_xj(self, xi, xj):
        if self.fixed == True:
            return 1 if self.pi_star.index(xi) < self.pi_star.index(xj) else 0
        
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