import numpy as np
from scipy import optimize


def llm_subsampling_bound(train_error, div, data_size, sample_size, delta = 1, epsilon=0.05):
    r = sample_size/(sample_size + data_size)
    complexity = np.sqrt((div - np.log(r * epsilon))/(2*data_size)) +np.sqrt(-np.log((1-r) * epsilon)/(2*sample_size))
    bound = train_error + delta * complexity

    print(f"train_error={train_error}")
    print(f"div={div}")
    print(f"data_size={data_size}")
    print(f"sample_size={sample_size}")
    print(f"delta={delta}")
    print(f"complexity={complexity}")
    print(f"bound={bound}")

    return bound 


def _bernoulli_inv_laplace(a, q):
    return np.expm1(-a * q) / np.expm1(-a)

def pac_bayes_bound(divergence, train_error, n, gamma, epsilon=0.05):
    """ This function computes a simple pac-bayes bound for generalization.

    Parameters
    ----------
    divergence: The KL-divergence between the posterior and the prior.
    train_error: The training error of the posterior.
    n: The number of samples.
    gamma: A hyperparameter trading off between the KL-divergence and the training error.
    epsilon: The probability with which the returned bound holds.

    Returns
    -------
    A 1 - epsilon probability upper bound on the testing error.
    """
    gamma = np.exp(gamma)

    q = train_error + (divergence - np.log(epsilon)) / gamma
    return _bernoulli_inv_laplace(gamma / n, q)


def compute_convexity_bound(train_error, div, sample_size, epsilon=0.05):
    def bound(lam):
        output = train_error / (1 - 0.5 * lam)
        num = div + np.log(1. / epsilon) + np.log(2. * np.sqrt(sample_size))
        denom = sample_size * lam * (1 - 0.5 * lam)
        output += num / denom
        return output

    a = 2. * sample_size * train_error
    b = div + np.log(1. / epsilon) + np.log(2. * np.sqrt(sample_size))
    lam = 2. / ((np.sqrt(a / b + 1.)) + 1.)
    result = bound(lam)

    return result


def compute_mcallester_bound(train_error, div, sample_size, epsilon=0.05):
    num = div - np.log(epsilon) + np.log(sample_size)  # Theta finite
    bound = train_error
    bound += np.sqrt(num / (2. * sample_size - 1.))
    return bound


def compute_catoni_bound(train_error, divergence, sample_size, epsilon=0.05, alpha=1e-4):
    log_eps = np.log(epsilon)
    inv_log_alpha = 1. / np.log1p(alpha)

    def bound_fn(g):
        q = train_error
        entropy = divergence - log_eps
        entropy += 2 * np.log(2 + np.log(g) * inv_log_alpha)
        q += ((1 + alpha) / g) * entropy
        return _bernoulli_inv_laplace(g / sample_size, q)

    result = optimize.minimize_scalar(bound_fn, (1., 100.))
    return result.fun