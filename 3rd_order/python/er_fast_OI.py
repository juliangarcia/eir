import itertools
from operator import itemgetter
import sys
import json
from math import fsum
from math import e
# from nakamura_otshuki import *
from otshuki_iwasa import *




def create_institution_space(rule_length=8):
    """
    Creates a list with all the rules, given a rule length
    :param rule_length:
    :return:
    """
    limit = 2 ** rule_length
    all_institutions = [None] * limit
    for institution_number in range(limit):
        all_institutions[institution_number] = list((map(int, "{0:b}".format(institution_number).zfill(rule_length))))
    return all_institutions


strategies = [(0, 0), (1, 1), (1, 0), (0, 1)]

institution_space = create_institution_space()


class Parameters:
    def __init__(self, R, S, T, P, population_size, intensity_of_selection, mutation_probability,
                 alpha, epsilon, chi, institution_code):
        self.R = R
        self.S = S
        self.T = T
        self.P = P
        self.population_size = population_size
        self.intensity_of_selection = intensity_of_selection
        self.mutation_probability = mutation_probability
        self.alpha = alpha
        self.epsilon = epsilon
        self.chi = chi
        self.institution_code = institution_code
        all_institutions = create_institution_space(8)
        self.d = np.array(all_institutions[institution_code])
        # APPLY ERRORS
        self.d = (1 - 2 * self.alpha) * self.d + self.alpha


def stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # builds a dictionary with position, eigenvalue
    # and retrieves from this, the index of the largest eigenvalue
    largest_eigenvalue_index = max(
        zip(range(0, len(eigenvalues)), eigenvalues), key=itemgetter(1))[0]
    # returns the normalized vector corresponding to the
    # index of the largest eigenvalue
    # and gets rid of potential complex values
    vector = np.real(eigenvectors[:, largest_eigenvalue_index])
    # normalise
    vector /= np.sum(vector, dtype=float)
    return vector


def D_p(h, h_prime, k, Z, p, X, X_bar):
    """
    Probability that P donates
    :param h:
    :param h_prime:
    :return:
    """
    first_term = (h / k) * (
    ((h - 1 + h_prime) / (Z - 1) * (C_G(p, X))) + ((Z - h - h_prime) / (Z - 1) * (C_B(p, X_bar))))
    second_term = ((k - h) / k) * (
    ((h + h_prime) / (Z - 1) * (C_G(p, X))) + (((Z - h - 1 - h_prime) / (Z - 1)) * (C_B(p, X_bar))))
    return first_term + second_term


def cooperation_index(param):

    #auxiliary for D computation
    X = np.array([1 - param.chi, param.chi])
    X_bar = np.array([param.chi, 1 - param.chi])


    #  first step is to compute the transition matrix of the moran process that describes evolution.
    evolution_matrix = np.zeros((4, 4))  # we have four strategies
    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                # chance that j appears in an i population
                evolution_matrix[i, j] = fixation_probability(j, i, param)
    evolution_matrix *= param.mutation_probability / 3
    for i in range(0, 4):
        evolution_matrix[i, i] = 1.0 - fsum(evolution_matrix[i, :])

    # now we compute the stationary distribution of the evolutionary dynamics

    stationary = stationary_distribution(evolution_matrix)

    space_reputation = list(
        itertools.product(range(param.population_size + 1), range(param.population_size - param.population_size + 1)))
    cooperation = 0.0
    reputation_dyn_monomorphous_per_strategy = [None, None, None, None]

    for i in range(4):
        inner = 0.0
        # aux = NakamuraOhtsuki(i, i, institution_code,
        #                      population_size, k=population_size, alpha=alpha, epsilon=epsilon, chi=chi)

        stationary_distribution_monomorphous = reputation_stationary_monomorphous(i, param)
        reputation_dyn_monomorphous_per_strategy[i] = stationary_distribution_monomorphous
        for j in range(param.population_size + 1):
            p_strategy = np.array(strategies[i])
            p_strategy = (1.0 - param.epsilon) * p_strategy
            D = D_p(j, 0, param.population_size, param.population_size, p_strategy, X, X_bar)
            index_state = space_reputation.index((j, 0))
            sigma = stationary_distribution_monomorphous[index_state]
            inner += (D * sigma)
        cooperation += (inner * stationary[i])
    return cooperation, stationary, space_reputation, reputation_dyn_monomorphous_per_strategy


def fixation_probability(mutant_index, resident_index, param):
    population_size = param.population_size
    intensity_of_selection = param.intensity_of_selection
    suma = np.zeros(population_size, dtype=np.float64)
    gamma = 1.0
    try:
        for i in range(1, population_size):
            strategies_vector = np.zeros(4, dtype=int)
            strategies_vector[mutant_index] = i
            strategies_vector[resident_index] = population_size - i
            payoff_mutant = payoff_function(mutant_index, population_composition=strategies_vector,
                                            param=param)
            payoff_resident = payoff_function(resident_index, population_composition=strategies_vector,
                                              param=param)
            fitness_mutant = e ** (intensity_of_selection * payoff_mutant)
            fitness_resident = e ** (intensity_of_selection * payoff_resident)
            gamma *= (fitness_resident / fitness_mutant)
            suma[i] = gamma
        return 1 / fsum(suma)
    except OverflowError:
        return 0.0














# REPUTATION STUFF BELOW



def probability(h_i, h_i_prime, h_j, h_j_prime, i, j, k, Z, p, p_prime, X, X_bar, d, output_gbar=None, output_GX=None):
    if h_j == h_i + 1 and h_j_prime == h_i_prime:
        return H_plus_p(h_i, h_i_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=output_gbar, output_GX=output_GX)
    elif h_j == h_i - 1 and h_j_prime == h_i_prime:
        return H_minus_p(h_i, h_i_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=output_gbar, output_GX=output_GX)
    elif h_j == h_i and h_j_prime == h_i_prime + 1:
        return H_plus_p_prime(h_i, h_i_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=output_gbar, output_GX=output_GX)
    elif h_j == h_i and h_j_prime == h_i_prime - 1:
        return H_minus_p_prime(h_i, h_i_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=output_gbar, output_GX=output_GX)
    elif i == j:
        return H_equal(h_i, h_i_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=output_gbar, output_GX=output_GX)
    else:
        return 0


def encounter_types(p, p_prime):
    encounter_types_vector = [None] * 8
    encounter_types_vector[0] = (p, p, 1, 1)
    encounter_types_vector[1] = (p, p, 1, 0)
    encounter_types_vector[2] = (p, p_prime, 1, 1)
    encounter_types_vector[3] = (p, p_prime, 1, 0)
    encounter_types_vector[4] = (p, p, 0, 1)
    encounter_types_vector[5] = (p, p, 0, 0)
    encounter_types_vector[6] = (p, p_prime, 0, 1)
    encounter_types_vector[7] = (p, p_prime, 0, 0)
    return encounter_types_vector


def encounter_probabilities(h, h_prime, k, Z):
    encounter_probabilities_vector = np.zeros(8)
    encounter_probabilities_vector[0] = h * (h - 1)
    encounter_probabilities_vector[1] = h * (k - h)
    encounter_probabilities_vector[2] = h * h_prime
    encounter_probabilities_vector[3] = h * (Z - k - h_prime)
    encounter_probabilities_vector[4] = h * (k - h)
    encounter_probabilities_vector[5] = (k - h) * (k - h - 1)
    encounter_probabilities_vector[6] = (k - h) * h_prime
    encounter_probabilities_vector[7] = (k - h) * (Z - k - h_prime)
    encounter_probabilities_vector /= k * (Z - 1)
    return encounter_probabilities_vector


def R_bar(focal_strategy, other_strategy, focal_rep, other_rep, X, X_bar):
    if other_rep == 1 and focal_rep == 1:
        return C_G(focal_strategy, X) * C_G(other_strategy, X)
    elif other_rep == 1 and focal_rep == 0:
        return C_G(focal_strategy, X) * C_B(other_strategy, X_bar)
    elif other_rep == 0 and focal_rep == 1:
        return C_B(focal_strategy, X_bar) * C_G(other_strategy, X)
    elif other_rep == 0 and focal_rep == 0:
        return C_B(focal_strategy, X_bar) * C_B(other_strategy, X_bar)


def S_bar(focal_strategy, other_strategy, focal_rep, other_rep, X, X_bar):
    if other_rep == 1 and focal_rep == 1:
        return C_G(focal_strategy, X) * (1.0 - C_G(other_strategy, X))
    elif other_rep == 1 and focal_rep == 0:
        return C_G(focal_strategy, X) * (1.0 - C_B(other_strategy, X_bar))
    elif other_rep == 0 and focal_rep == 1:
        return C_B(focal_strategy, X_bar) * (1.0 - C_G(other_strategy, X))
    elif other_rep == 0 and focal_rep == 0:
        return C_B(focal_strategy, X_bar) * (1.0 - C_B(other_strategy, X_bar))


def P_bar(focal_strategy, other_strategy, focal_rep, other_rep, X, X_bar):
    if other_rep == 1 and focal_rep == 1:
        return (1.0 - C_G(focal_strategy, X)) * (1.0 - C_G(other_strategy, X))
    elif other_rep == 1 and focal_rep == 0:
        return (1.0 - C_G(focal_strategy, X)) * (1.0 - C_B(other_strategy, X_bar))
    elif other_rep == 0 and focal_rep == 1:
        return (1.0 - C_B(focal_strategy, X_bar)) * (1.0 - C_G(other_strategy, X))
    elif other_rep == 0 and focal_rep == 0:
        return (1.0 - C_B(focal_strategy, X_bar)) * (1.0 - C_B(other_strategy, X_bar))


def T_bar(focal_strategy, other_strategy, focal_rep, other_rep, X, X_bar):
    if other_rep == 1 and focal_rep == 1:
        return (1.0 - C_G(focal_strategy, X)) * C_G(other_strategy, X)
    elif other_rep == 1 and focal_rep == 0:
        return (1.0 - C_G(focal_strategy, X)) * C_B(other_strategy, X_bar)
    elif other_rep == 0 and focal_rep == 1:
        return (1.0 - C_B(focal_strategy, X_bar)) * C_G(other_strategy, X)
    elif other_rep == 0 and focal_rep == 0:
        return (1.0 - C_B(focal_strategy, X_bar)) * C_B(other_strategy, X_bar)


def R_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar):
    return np.dot(np.array([R_bar(*my_tuple, X, X_bar) for my_tuple in encounter_types_vector]),
                  encounter_probabilities_vector)


def S_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar):
    return np.dot(np.array([S_bar(*my_tuple, X, X_bar) for my_tuple in encounter_types_vector]),
                  encounter_probabilities_vector)


def T_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar):
    return np.dot(np.array([T_bar(*my_tuple, X, X_bar) for my_tuple in encounter_types_vector]),
                  encounter_probabilities_vector)


def P_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar):
    return np.dot(np.array([P_bar(*my_tuple, X, X_bar) for my_tuple in encounter_types_vector]),
                  encounter_probabilities_vector)


def reputation_stationary_monomorphous(p_index, param):
    # this is mostly copied so may need refactoring at some point
    Z = param.population_size
    k = Z
    institution_code = param.institution_code
    alpha = param.alpha
    reputation_space_size = (k + 1) * (Z - k + 1)

    d = np.array(institution_space[institution_code])
    d = (1 - 2 * alpha) * d + alpha

    p = np.array(strategies[p_index])
    p_prime = np.array(strategies[p_index])

    # APPLY ERRORS
    p = (1.0 - param.epsilon) * p
    p_prime = (1.0 - param.epsilon) * p_prime
    X = np.array([1 - param.chi, param.chi])
    X_bar = np.array([param.chi, 1 - param.chi])

    space = list(itertools.product(range(k + 1), range(Z - k + 1)))

    # build the matrix of the reputation dynamics - this is the part that needs to change for the reputational space

    reputation_dynamics_matrix = np.zeros((reputation_space_size, reputation_space_size))
    #GX_out = np.empty((2, 4))
    # gbar_out = np.empty((2, 2))
    GX_out = np.empty(8)
    gbar_out = np.empty(4)

    for i in range(reputation_space_size):
        for j in range(reputation_space_size):
            h_i, h_i_prime = space[i]  # start_state
            h_j, h_j_prime = space[j]  # destination state
            # TODO: add parameters
            reputation_dynamics_matrix[i, j] = probability(h_i, h_i_prime, h_j, h_j_prime, i, j, k, Z, p, p_prime, X,
                                                           X_bar, d, output_gbar=gbar_out, output_GX=GX_out)

    # compute the stationary distribution of the reputation dynamics
    return stationary_distribution(reputation_dynamics_matrix)


def payoff_function(index, population_composition, param):
    assert type(population_composition) == np.ndarray, "Composition is not an array"
    alpha = param.alpha
    epsilon = param.epsilon
    chi = param.chi
    R = param.R
    S = param.S
    T = param.T
    P = param.P
    institution_code = param.institution_code
    support = np.nonzero(population_composition)[0]
    assert len(support) == 2, "This function only works for dimorphic populations"
    if index == support[0]:
        p_prime_strategy_index = support[1]
    elif index == support[1]:
        p_prime_strategy_index = support[0]
    else:
        raise ValueError("Something is wrong with the parameters")
    Z = np.sum(population_composition)
    k = population_composition[index]
    reputation_space_size = (k + 1) * (Z - k + 1)

    d = np.array(institution_space[institution_code])
    d = (1 - 2 * alpha) * d + alpha

    p = np.array(strategies[index])
    p_prime = np.array(strategies[p_prime_strategy_index])

    # APPLY ERRORS
    p = (1.0 - epsilon) * p
    p_prime = (1.0 - epsilon) * p_prime
    X = np.array([1 - chi, chi])
    X_bar = np.array([chi, 1 - chi])

    space = list(itertools.product(range(k + 1), range(Z - k + 1)))

    # build the matrix of the reputation dynamics - this is the part that needs to change for the reputational space

    reputation_dynamics_matrix = np.zeros((reputation_space_size, reputation_space_size))
    GX_out = np.empty(8)
    gbar_out = np.empty(4)

    for i in range(reputation_space_size):
        for j in range(reputation_space_size):
            h_i, h_i_prime = space[i]  # start_state
            h_j, h_j_prime = space[j]  # destination state

            reputation_dynamics_matrix[i, j] = probability(h_i, h_i_prime, h_j, h_j_prime, i, j, k, Z, p, p_prime, X,
                                                           X_bar, d, output_gbar=gbar_out, output_GX=GX_out)

    # compute the stationary distribution of the reputation dynamics
    stat_dist = stationary_distribution(reputation_dynamics_matrix)

    # now the payoff
    p_payoff_value = 0.0
    encounter_types_vector = encounter_types(p, p_prime)
    for i, state in enumerate(space):
        h, h_prime = state
        encounter_probabilities_vector = encounter_probabilities(h, h_prime, k, Z)
        p_payoff = R * R_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar) + \
                   S * S_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar) + \
                   T * T_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar) + \
                   P * P_p(h, h_prime, encounter_types_vector, encounter_probabilities_vector, X, X_bar)
        p_payoff_value += p_payoff * stat_dist[i]
    return p_payoff_value


def inspect(mutant_index, number_of_mutants, resident_index, param):
    population_size = param.population_size
    number_of_residents = population_size - number_of_mutants
    strategies_vector = np.zeros(4, dtype=int)
    strategies_vector[mutant_index] = number_of_mutants
    strategies_vector[resident_index] = number_of_residents
    payoff_mutant = payoff_function(mutant_index, population_composition=strategies_vector,
                                    param=param)
    payoff_resident = payoff_function(resident_index, population_composition=strategies_vector,
                                      param=param)
    fitness_mutant = e ** (param.intensity_of_selection * payoff_mutant)
    fitness_resident = e ** (param.intensity_of_selection * payoff_resident)

    # now reputation dynamics
    alpha = param.alpha
    epsilon = param.epsilon
    chi = param.chi
    R = param.R
    S = param.S
    T = param.T
    P = param.P
    institution_code = param.institution_code
    support = np.nonzero(strategies_vector)[0]
    assert len(support) == 2, "This function only works for dimorphic populations"
    index = mutant_index
    p_prime_strategy_index = resident_index
    Z = number_of_mutants + number_of_residents
    k = number_of_mutants
    reputation_space_size = (k + 1) * (Z - k + 1)

    d = np.array(institution_space[institution_code])
    d = (1 - 2 * alpha) * d + alpha

    p = np.array(strategies[index])
    p_prime = np.array(strategies[p_prime_strategy_index])

    # APPLY ERRORS
    p = (1.0 - epsilon) * p
    p_prime = (1.0 - epsilon) * p_prime
    X = np.array([1 - chi, chi])
    X_bar = np.array([chi, 1 - chi])

    space = list(itertools.product(range(k + 1), range(Z - k + 1)))

    # build the matrix of the reputation dynamics - this is the part that needs to change for the reputational space

    reputation_dynamics_matrix = np.zeros((reputation_space_size, reputation_space_size))
    GX_out = np.empty(8)
    gbar_out = np.empty(4)

    for i in range(reputation_space_size):
        for j in range(reputation_space_size):
            h_i, h_i_prime = space[i]  # start_state
            h_j, h_j_prime = space[j]  # destination state

            reputation_dynamics_matrix[i, j] = probability(h_i, h_i_prime, h_j, h_j_prime, i, j, k, Z, p, p_prime, X,
                                                           X_bar, d, output_gbar=gbar_out, output_GX=GX_out)

    # compute the stationary distribution of the reputation dynamics
    stat_dist = stationary_distribution(reputation_dynamics_matrix)

    #prepare output
    ans = dict()
    ans['p_mutant'] = payoff_mutant
    ans['p_resident'] = payoff_resident
    ans['fitness_mutant'] = fitness_mutant
    ans['fitness_resident'] = fitness_resident
    ans['stationary_reputation'] = list(stat_dist)
    #ans['rep_dynamics'] = reputation_dynamics_matrix
    return ans



# MAIN FUNCTION


def main(argv=sys.argv):
    param = Parameters(R=float(argv[3]), S=float(argv[4]), T=float(argv[5]), P=float(argv[6]),
                       population_size=int(argv[2]), intensity_of_selection=1.0, mutation_probability=0.01,
                       alpha=0.01, epsilon=0.08, chi=0.01, institution_code=int(argv[1]))

    # COMPUTE INDEX
    index, stationary, space_reputation, reputation_dyn_monomorphous_per_strategy = cooperation_index(param)

    # PREPARE OUTPUT
    ans = dict()
    ans["alpha"] = param.alpha
    ans["epsilon"] = param.epsilon
    ans["chi"] = param.chi
    ans["R"] = param.R
    ans["S"] = param.S
    ans["T"] = param.T
    ans["P"] = param.P
    ans["index"] = index
    ans["institution"] = param.institution_code
    ans["population_size"] = param.population_size

    #extra
    ans["stationary_strategies"] = stationary.tolist()
    ans["space_reputation"] = space_reputation
    ans["reputation_dynamics_0"] = reputation_dyn_monomorphous_per_strategy[0].tolist()
    ans["reputation_dynamics_1"] = reputation_dyn_monomorphous_per_strategy[1].tolist()
    ans["reputation_dynamics_2"] = reputation_dyn_monomorphous_per_strategy[2].tolist()
    ans["reputation_dynamics_3"] = reputation_dyn_monomorphous_per_strategy[3].tolist()
    #NO
    name = "OI_inst_{}_n_{}_R_{}_S_{}_T_{}_P_{}.json".format(param.institution_code, param.population_size, param.R,
                                                             param.S,
                                                             param.T, param.P)
    with open(name, 'w') as fp:
        json.dump(ans, fp)

if __name__ == "__main__":
    # PARSE ARGUMENTS
    main()

