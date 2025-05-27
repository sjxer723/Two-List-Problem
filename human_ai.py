import itertools
from model.distribution import Distribution

def utilities_joint_system(Ds:list[Distribution], selected_items, u):
    """
    Calculate the utilities of each human in the joint system
    Args:
        Ds (list): the set of noisy distributions of all types of humans
        selected_items (list): the set of presented items by the algorithm
        u (list): the utility function
    Return:
        joint_system_utilities: utilities of each type of humans
    """
    joint_system_utilities = []
    for _, D in enumerate(Ds):
        central_ranking = D.pi_star
        utility = 0
        for g in selected_items:
            prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
            idx_of_g = central_ranking.index(g)
            utility += u[idx_of_g] * prob_selecting_g
        joint_system_utilities.append(utility)

    return joint_system_utilities

def utility_joint_system(D:Distribution, selected_items, u):
    """
    Calculate the utility of a single type of human
    Args:
        D: the noisy distribution of a single type of human
        selected_items (list): the set of presented items by the algorithm
        u (list): the utility function
    Return:
        utility: the utility of the human
    """
    central_ranking = D.pi_star
    utility = 0
    for g in selected_items:
        prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
        idx_of_g = central_ranking.index(g)
        utility += u[idx_of_g] * prob_selecting_g

    return utility

def utilities_joint_system1(Ds:list[Distribution], D_a:Distribution, k, u):
    """
    Calculate the utilities of each human in the joint system when the algorithm is also noisy
    Args:
        Ds (list): the set of noisy distributions of all types of humans
        D_a: the noisy distribution of the algorithm
        k (int): the number of presented items
        u (list): the utility function
    Return:
        joint_system_utilities: utilities of each type of humans
    """
    joint_system_utilities = [0 for _ in range(len(Ds))]

    # enumerate all the possible set of presented items
    for selected_items in itertools.combinations(D_a.pi_star, k):
        prob_selecting = D_a.prob_of_fixed_unordered_prefix(list(selected_items))
        for i, D in enumerate(Ds):
            central_ranking = D.pi_star
            for g in selected_items:
                prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
                idx_of_g = central_ranking.index(g)
                joint_system_utilities[i] += u[idx_of_g] * prob_selecting_g * prob_selecting

    return joint_system_utilities

def utility_joint_system1(D:Distribution, D_a:Distribution, k, u):
    """
    Calculate the utility of a single type of human when the algorithm is also noisy
    Args:
        D: the noisy distribution of a single type of human
        D_a: the noisy distribution of the algorithm
        k (int): the number of presented items
        u (list): the utility function
    Return:
        utility: the utility of the human
    """
    utility = 0
    for selected_items in itertools.combinations(D_a.pi_star, k):
        prob_selecting = D_a.prob_of_fixed_unordered_prefix(list(selected_items))
        central_ranking = D.pi_star
        for g in selected_items:
            prob_selecting_g = D.prob_of_xi_before_S(g, selected_items)
            idx_of_g = central_ranking.index(g)
            utility += u[idx_of_g] * prob_selecting_g * prob_selecting

    return utility


def utilities_human(Ds:list[Distribution], u):
    """
    Calculate the utilities of each human when acting alone
    Args:
        Ds (list): the set of noisy distributions of all types of humans
        u (list): the utility function
    Return:
        human_utilities (list): utilities of each type of humans
    """
    human_utilities = []
    for _, D in enumerate(Ds):
        central_ranking = D.pi_star
        utility = 0
        for idx, g in enumerate(central_ranking):
            prob_selecting_g = D.prob_of_xi_being_first_k(g, 1)
            utility += prob_selecting_g * u[idx]
        human_utilities.append(utility)

    return human_utilities
