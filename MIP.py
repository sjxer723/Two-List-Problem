from gurobipy import Model, GRB, LinExpr
from utils import *

class StrategyMIP():
    def __init__(self, m, freqs, Ds, k, u, verbose=False):
        self.m = m                  # number of items
        self.k = k                  # number of items to be presented
        self.freqs = freqs          # frequency of each type of human
        self.Ds = Ds                # Mallows models
        self.types = len(freqs)     # number of types of human
        self.u = u                  # utility function  
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
        """
        Reset the frequency of each type of human
        """
        self.freqs = freqs

    def create_vars(self, integer=False):
        """
        Create variables for the model
        Args:
            integer (bool): whether to set the first m variables as integer, if False, the program turns to a LP
        """
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
        Solve the MIP to find the optimal 
        Args:
            uplift (bool): whether to add the uplift constraint
            human_utility (float): utility of the human when acting alone (only used when uplift is True)
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


        if uplift: # Add uplift constraint
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