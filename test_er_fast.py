import numpy as np
from er_fast_OI import gbar
from er_fast_OI import G_G
from er_fast_OI import G_B
from er_fast_OI import cooperation_index
from er_fast_OI import Parameters



def test_gbar_with_out():
    #out = np.zeros((2, 2))
    out = np.zeros(4)
    p = np.array([0.92, 0.92])
    p_prime = np.array([0.92, 0.92])
    one_or_zero_a = 1
    one_or_zero_b = 0
    X = np.array([0.99, 0.01])
    X_bar = np.array([0.01, 0.99])
    result = np.array(gbar(p, p_prime, one_or_zero_a, one_or_zero_b, X, X_bar, output_gbar=out))
    expected = np.array([ 0.8464,  0.0736,  0.0736,  0.0064])
    np.testing.assert_almost_equal(result, expected)

def test_G_G_with_out():
    #out_vec_GG = np.zeros((2, 4))
    out_vec_GG = np.zeros(8)
    #out_gbar = np.zeros((2, 2))
    out_gbar = np.zeros(4)
    p = np.array([0.92, 0.92])
    p_prime = np.array([0.92, 0.92])
    X = np.array([0.99, 0.01])
    X_bar = np.array([0.01, 0.99])
    one_or_zero = 1
    d = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.99])
    result = np.array(G_G(p, p_prime, one_or_zero, X, X_bar, d, output_gbar=out_gbar, output_GG=out_vec_GG))
    expected = 0.010784
    np.testing.assert_almost_equal(result, expected)





def test_G_B_with_out():
    p = np.array([0.92, 0.92])
    p_prime = np.array([0.92, 0.92])
    X = np.array([0.99, 0.01])
    X_bar = np.array([0.01, 0.99])
    one_or_zero = 1
    d = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.99])
    #out_vec_GG = np.zeros((2, 4))
    #out_gbar = np.zeros((2, 2))
    out_vec_GG = np.zeros(8)
    out_gbar = np.zeros(4)
    result = G_B(p, p_prime, one_or_zero, X, X_bar, d, output_gbar=out_gbar, output_GB=out_vec_GG)
    expected = 0.087615999999999958
    np.testing.assert_almost_equal(result, expected)


def test_index_nakamura():
    param = Parameters(R=3.0, S=0.0, T=5.0, P=1.0, population_size=10, intensity_of_selection=1.0, mutation_probability=0.01,
                       alpha=0.01, epsilon=0.08, chi=0.01, institution_code=5)
    # COMPUTE INDEX
    index, stationary, space_reputation, reputation_dyn_monomorphous_per_strategy = cooperation_index(param)
    np.testing.assert_almost_equal(index, 0.0007806)

