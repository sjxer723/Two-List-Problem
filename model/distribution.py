class Distribution(object):
    def __init__(self):
        raise NotImplementedError

    def items() -> list:
        raise NotImplementedError

    def sample(self) -> list:
        raise NotImplementedError
    
    def prob_of_fixed_unordered_prefix(self, items):
        raise NotImplementedError

    def prob_of_xi_before_xj(self, xi, xj):
        raise NotImplementedError