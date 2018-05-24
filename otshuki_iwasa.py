import numpy as np
import numba


######

@numba.njit
def outer_numba(a, b, result):
    m = a.shape[0]
    n = b.shape[0]
    for i in range(m):
        for j in range(n):
            result[i*n + j] = a[i] * b[j]
    return result

def C_G(p, X):
    return np.dot(X, p)


def C_B(p, X_bar):
    return np.dot(X_bar, p)


def cbar(p, one_or_zero, X, X_bar):
    if one_or_zero == 0:
        return np.array([C_B(p, X_bar), 1 - C_B(p, X_bar)])
    elif one_or_zero == 1:
        return np.array([C_G(p, X), 1 - C_G(p, X)])
    else:
        raise ValueError("I should never be here.")

# @numba.njit
def gbar(p, p_prime, one_or_zero_a, one_or_zero_b, X, X_bar, output_gbar=None):
    return outer_numba(cbar(p, one_or_zero_b, X, X_bar), cbar(p_prime, one_or_zero_a, X, X_bar), output_gbar)


###########


# @numba.njit
#def G_G(p, p_prime, one_or_zero, X, X_bar, d, output_gbar=None, output_GG=None):
#    return np.dot(outer_numba(X, gbar(p, p_prime, one_or_zero, 1, X, X_bar, output_gbar=output_gbar), output_GG), d)

#output_gbar is 2x2
#output_GG 4x2
def G_GG(p, X, d, output_gbar=None, output_GG=None):
    C_vector = np.array([C_G(p, X), 1 - C_G(p, X)])
    kron = outer_numba(X, outer_numba(X, C_vector, result=output_gbar), result=output_GG)
    return np.dot(kron, d)


def G_GB(p, X, X_bar, d, output_gbar=None, output_GG=None):
    C_vector = np.array([C_B(p, X_bar), 1 - C_B(p, X_bar)])
    a = outer_numba(X, outer_numba(X_bar, C_vector, result=output_gbar), result=output_GG)
    return np.dot(a, d)





# @numba.njit
#def G_B(p, p_prime, one_or_zero, X, X_bar, d, output_gbar=None, output_GB=None):
#    return np.dot(outer_numba(X_bar, gbar(p, p_prime, one_or_zero, 0, X, X_bar, output_gbar=output_gbar), output_GB), d)

def G_BG(p, X, X_bar, d, output_gbar=None, output_GB=None):
    C_vector = np.array([C_G(p, X), 1 - C_G(p, X)])
    a = outer_numba(X_bar, outer_numba(X, C_vector, result=output_gbar), result=output_GB)
    return np.dot(a, d)


def G_BB(p, X_bar, d, output_gbar=None, output_GB=None):
    C_vector = np.array([C_B(p, X_bar), 1 - C_B(p, X_bar)])
    a = outer_numba(X_bar, outer_numba(X_bar, C_vector, result=output_gbar), result=output_GB)
    return np.dot(a, d)


def H_plus_p(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=None, output_GX=None):
    factor = (k - h) / Z
    term1 = ((h + h_prime) * G_BG(p, X, X_bar, d, output_gbar, output_GX) / (Z - 1))
    term2 = ((Z - h - h_prime - 1) * G_BB(p, X_bar, d, output_gbar, output_GX) / (Z - 1))
    value = term1 + term2
    return factor * value


def H_minus_p(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=None, output_GX=None):
    factor = h / Z
    term1 = (h + h_prime - 1) * (1 - G_GG(p, X, d, output_gbar, output_GX)) / (Z - 1)
    term2 = (Z - h - h_prime) * (1 - G_GB(p, X, X_bar, d, output_gbar, output_GX)) / (Z - 1)
    value = term1 + term2
    return factor * value


def H_minus_p_prime(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=None, output_GX=None):
    factor = h_prime / Z
    term1 = (h + h_prime - 1) * (1 - G_GG(p_prime, X, d, output_gbar, output_GX)) / (Z - 1)
    term2 = (Z - h - h_prime) * (1 - G_GB(p_prime, X, X_bar, d, output_gbar, output_GX)) / (Z - 1)
    value = term1 + term2
    return factor * value


def H_plus_p_prime(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=None, output_GX=None):
    factor = (Z - k - h_prime) / Z
    value = (h + h_prime) * G_BG(p_prime, X, X_bar, d, output_gbar, output_GX) / (Z - 1) + \
            (Z - h - h_prime - 1) * G_BB(p_prime, X_bar, d, output_gbar, output_GX) / (Z - 1)
    return factor * value


def H_equal(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar=None, output_GX=None):
    return 1 - H_plus_p(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar, output_GX) \
           - H_minus_p(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar, output_GX) \
           - H_plus_p_prime(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar, output_GX) \
           - H_minus_p_prime(h, h_prime, k, Z, p, p_prime, X, X_bar, d, output_gbar, output_GX)
