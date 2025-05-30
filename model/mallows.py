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
        self.Z_lam = lambda l: sum(self.Z_lst[:l]) if l > 0 else 0
        self.Z_neg_lam = lambda l, k: sum(self.Z_lst[k - l: k])
        self.Z = 1
        for i in range(_m):
            self.Z *= self.Z_lam(i + 1)
        self.fixed = _fixed
    
    def items(self) -> list:
        return self.pi_star
    
    def kt_distance(self, permutation):
        """
        Calculate the Kendall tau distance between the permutation and the model's pi_star.
        """
        if self.fixed == True:
            return 0
        if len(permutation) != self.m:
            raise ValueError("Length of permutation must be equal to m")
        
        # Count inversions
        inversions = 0
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if self.pi_star.index(permutation[i]) > self.pi_star.index(permutation[j]):
                    inversions += 1
        return inversions
    
    def prob(self, permutation) -> float:
        """
        Calculate the probability of a given permutation under the Mallows model.
        """
        if self.fixed == True:
            return 1 if permutation == self.pi_star else 0
        if len(permutation) != self.m:
            raise ValueError("Length of permutation must be equal to m")
        
        # Count inversions
        inversions = self.kt_distance(permutation)
        
        # Calculate the probability
        prob = np.exp(-self.phi * inversions)
        return prob / self.Z 

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
    
    def prob_of_xi_before_S(self, xi, S=None, verbose=False):
        if S is None:
            S = self.items()
        if xi not in S:
            S = [xi] + S
        if verbose:
            print("xi: ", xi)
            print("S: ", S)

        # dp[i, j, k]: probability of xi being chosen at position s after the k-th step
        dp = np.zeros((self.m + 1, self.m + 1, self.m + 1))        
        for k in range(1, self.m + 1):
            for i in range(1, k):
                for s in range(1, k+1):
                    # gamma_{k, s}
                    gamma = self.Z_neg_lam(s, k) / self.Z_lam(k)
                    # gamma_{k, s-1}
                    gamma_prime = self.Z_neg_lam(s - 1, k) / self.Z_lam(k) 
                    dp[i][s][k] = (1 - gamma) * dp[i][s][k - 1] + (gamma_prime * dp[i][s-1][k - 1] if self.pi_star[k - 1] not in S else 0)                    
            for s in range(1, k + 1):
                prod = 1
                for i in range(1, k):
                    prod *= self.pi_star[i - 1] not in S
                if self.pi_star[k - 1] in S:
                    sum_of_dp = 0
                    for i in range(1, k):
                        for l in range(s, self.m + 1):
                            sum_of_dp += dp[i][l][k - 1]
                    # if k is in S, we can only choose it at this position
                    dp[k][s][k] = self.Z_lst[k - s] / self.Z_lam(k) * (sum_of_dp + prod)
                else:
                    dp[k][s][k] = 0

        prob_of_xi = sum([dp[self.pi_star.index(xi) + 1][s][self.m] for s in range(1, self.m + 1)])
        normal_prob = [sum([dp[self.pi_star.index(xj) + 1][s][self.m] for s in range(1, self.m + 1)]) for xj in S]
        
        if verbose:
            print("prob_of_xi: ", prob_of_xi)
            print("normal_prob: ", normal_prob)
            print("sum(normal_prob): ", sum(normal_prob))

        return prob_of_xi / sum(normal_prob)
    
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

    def prob_of_xi_being_first_k(self, xi, k, verbose=False):
        items_except_xi = [item for item in self.pi_star if item != xi]
        index_of_xi = self.pi_star.index(xi) + 1
        prob_prefix = np.exp(-self.phi * (index_of_xi - k))
        prob_prefix *= math.prod([self.Z_lam(i + 1) / self.Z_lam(self.m - i) for i in range(k)])

        dp = np.zeros((self.m, k + 1))  # dp[i][j]: recursive_sum(i, j) for first i items with j selected 
        
        # Base cases
        for i in range(self.m):
            dp[i][0] = 1
        for j in range(k + 1):
            dp[0][j] = 0  # recursive_sum(0, k_prime) = 0 for k_prime > 0

        for num_of_remainings in range(1, self.m):
            for k_prime in range(1, k + 1):
                if num_of_remainings < k_prime:
                    dp[num_of_remainings][k_prime] = 0
                elif num_of_remainings == k_prime:
                    sum_of_0_to_k_prime_minus_1 = sum([i for i in range(k_prime)])
                    sum_of_indices = sum([self.pi_star.index(items_except_xi[i]) for i in range(num_of_remainings)])
                    dp[num_of_remainings][k_prime] = np.exp(- self.phi * (sum_of_indices - sum_of_0_to_k_prime_minus_1))
                else:
                    last_item = items_except_xi[num_of_remainings - 1]
                    prob_prefix1 = np.exp(- self.phi * (self.pi_star.index(last_item) + 1 - k_prime))
                    dp[num_of_remainings][k_prime] = prob_prefix1 * dp[num_of_remainings - 1][k_prime - 1] + dp[num_of_remainings - 1][k_prime]

        if verbose:
            print("DP result for remaining items:", dp[self.m - 1][k - 1])

        return prob_prefix * dp[self.m - 1][k - 1]
            
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