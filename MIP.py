import numpy as np
from gurobipy import Model, GRB, LinExpr
from model.mallows import Mallows
from utils import *

class StrategyMIP():
    def __init__(self, m, k, u, phi=1):
        self.m = m                  # number of items
        self.k = k                  # number of items to be presented
        self.D = Mallows(m, phi)    # Mallows model
        self.u = u
        self.W_num = m * m * m      # W(i, s, t): the probability of item i at position s after the t-th step
        self.y_num = m * m * m
        self.z_num = m * m
        self.q_num = m
        self.indicator_num = m
        self.var_num = self.indicator_num + self.W_num + self.y_num + self.z_num + self.q_num
        self.model = Model()        # Create Gurobi model
        
        # Create variables
        self.vars = {}
        for i in range(self.var_num):
            if i <= self.m:
                self.vars[i] = self.model.addVar(vtype=GRB.INTEGER, name=f"var_{i}")
            else:
                self.vars[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"var_{i}")
        # for i in range(self.var_num):
        #     self.vars[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"var_{i}")
        
        self.model.update()

    def reset_u(self, u):
        self.u = u

    def x(self, i):
        return (i - 1)
    
    def W(self, i, s, t):
        return self.indicator_num + ((i - 1) * self.m * self.m) + ((s - 1) * self.m) + (t - 1)
    
    def y(self, i, s, t):
        return self.indicator_num + self.W_num + ((i - 1) * self.m * self.m) + ((s - 1) * self.m) + (t - 1)
    
    def z(self, s, t):
        return self.indicator_num + self.W_num + self.y_num + ((s - 1) * self.m) + (t - 1) 

    def q(self, t):
        return self.indicator_num + self.W_num + self.y_num + self.z_num + (t - 1)

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

    def solve(self):
        """
            Solve the MIP to find the optimal strategy
        """
        for t in range(1, self.m + 1):
            for i in range(1, t):
                for s in range(1, t+1):
                    # gamma_{t, s}
                    gamma = self.D.Z_neg_lam(s, t) / self.D.Z_lam(t)
                    self.add_eq([self.W(i, s, t), self.W(i, s, t-1), self.y(i, s, t)], [1, gamma-1, -1])
                    # gamma_{t, s-1}
                    gamma_prime = self.D.Z_neg_lam(s - 1, t) / self.D.Z_lam(t) 
                    # y[i, s, t] <= gamma_{t, s-1} * W[i, s-1, t-1]
                    self.add_leq([self.y(i, s, t), self.W(i, s-1, t-1)], [1, -gamma_prime], 0)
                    # y[i, s, t] <= gamma_{t, s-1} * (1 - x[i])
                    self.add_leq([self.y(i, s, t), self.x(i)], [1, gamma_prime], gamma_prime)

            for s in range(1, t + 1):
                # W[t, s, t] = z[s, t]
                self.add_eq([self.W(t, s, t), self.z(s, t)], [1, -1])

                # z[s, t] <= p_{t, s}
                w_idxs = []
                for i in range(1, t):
                    for l in range(s, self.m + 1):
                        w_idxs.append(self.W(i, l, t-1))
                p_ts = self.D.Z_lst[t - s] / self.D.Z_lam(t)
                if t == 1:
                    self.add_leq([self.z(s, t)] + w_idxs, [1] + [-p_ts]*len(w_idxs), p_ts)
                else:
                    self.add_leq([self.z(s, t)] + w_idxs + [self.q(t-1)], [1] + [-p_ts]*len(w_idxs) + [-p_ts], 0)
                    
                # 0 <= z[s, t] <= p_{t, s} * x[t]
                self.add_leq([self.z(s, t), self.x(t)], [1, -p_ts], 0)
                self.add_geq([self.z(s, t)], [1], 0)

        for t in range(1, self.m):
            for i in range(1, t+1):
                # q[t] + x[i] <= 1
                self.add_leq([self.q(t), self.x(i)], [1, 1], 1)
                # q[t] + x[i] >= 0
                self.add_geq([self.q(t), self.x(i)], [1, 1], 0)
            
            # q[t] + x[1] + x[2] + ... + x[t] >= 1
            # self.add_geq([self.q(t)] + [self.x(i) for i in range(1, t + 1)], [1] * (t + 1), 1)

        # 0 <= x[i] <= 1
        self.add_binary([self.x(i) for i in range(1, self.indicator_num + 1)])
        # x[1] + x[2] + ... + x[m] <= k
        self.add_leq([self.x(i) for i in range(1, self.indicator_num+1)], [1] * self.indicator_num, self.k)


        # Set the objective
        objective_expr = LinExpr()
        for i in range(1, self.m+1):
            for s in range(1, self.m+1):
                objective_expr.addTerms(self.u[i-1], self.vars[self.W(i, s, self.m)])
        self.model.setObjective(objective_expr, GRB.MAXIMIZE)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            solution = [v.x for v in self.model.getVars()]
            return solution[:self.m]
        else:
            raise RuntimeError("Failed to solve the MIP: " + str(self.model.status))

m = 9

# u1 = [1 if i < 3 else 0 for i in range(m)]
# u2 = [i for i in range(m, 0, -1)]
# u3 = [np.exp(-i*0.1) for i in range(m)]

u1 = [1, 0, 1, 1, 0, 0, 0, 0]
StrategyMIP_instance = StrategyMIP(m=8, k=2, u=u1)
sol1 = StrategyMIP_instance.solve()
# StrategyMIP_instance.reset_u(u2)
# sol2 = StrategyMIP_instance.solve()
# StrategyMIP_instance.reset_u(u3)
# sol3 = StrategyMIP_instance.solve()


info("u1 = {}".format(u1))
print("Optimal solution:", sol1)
# info("u2 = {}".format(u2))
# print("Optimal solution:", sol2)
# info("u3= {}".format(u3))
# print("Optimal solution:", sol3)

