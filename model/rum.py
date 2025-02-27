import random
import numpy as np
from model.distribution import Distribution

class SuperStar(Distribution):
    def __init__(self, _m, _super_star, _p_sigmas):
        self.m = _m
        self.super_star = _super_star
        if len(_p_sigmas) != _m:
            raise ValueError("Length of p_sigma must be equal to m")
        self.p_sigmas = _p_sigmas / np.sum(_p_sigmas)

    def items(self) -> list:
        return list(range(1, self.m + 1))
    
    def sample(self) -> list:
        """
        Sample a permutation from the SuperStar model.
        """
        permutation = self.items()
        permutation.remove(self.super_star)
        random.shuffle(permutation)

        weights = self.p_sigmas
        weights /= np.sum(weights)
        r = np.random.choice(self.m, p=weights)
        permutation.insert(r, self.super_star)
        
        return permutation

    def prob_of_fixed_unordered_prefix(self, items):
        """
        The probability of the prefix is specified:
        probability of pi[:k] = {item[0], item[1], ..., item[k-1]} (in an arbitrary order)
        """
        k = len(items)
        if self.super_star not in items:
            prob_of_fixed_prefix = 0
            for i in range(k, self.m):
                prob_of_fixed_prefix_with_star_being_i = 1.0
                prob_of_fixed_prefix_with_star_being_i *= self.p_sigmas[i]
                for j in range(k):
                    prob_of_fixed_prefix_with_star_being_i *= (j + 1) * 1.0 / (self.m - 1 - j)
                prob_of_fixed_prefix += prob_of_fixed_prefix_with_star_being_i

            return prob_of_fixed_prefix
        else:
            prob_of_fixed_prefix = sum(self.p_sigmas[:k])
            for i in range(k - 1):
                prob_of_fixed_prefix *= (i + 1) * 1.0 / (self.m - 1 - i)
            return prob_of_fixed_prefix           
        
    
    def prob_of_xi_before_xj(self, xi, xj):
        if xi != self.super_star and xj != self.super_star:
            return 0.5
        elif xi == self.super_star and xj != self.super_star:
            prob_of_star_before_xj = 0
            for i in range(self.m):
                prob_of_star_before_xj += self.p_sigmas[i] * (self.m - i - 1) / (self.m - 1)
            return prob_of_star_before_xj
        elif xi != self.super_star and xj == self.super_star:
            prob_of_xi_before_star = 0
            for i in range(self.m):
                prob_of_xi_before_star += self.p_sigmas[i] * i / (self.m - 1)
            return prob_of_xi_before_star

        return 0 