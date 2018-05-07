import autograd.numpy as npa
from autograd import hessian_vector_product as hvp, value_and_grad
from autograd.core import primitive
from numpy.random import normal
from scipy.optimize import minimize

gaussian_numbers = normal(size=1000)
n_instances = 0
n_labels = 0
permutations = 0
tau = 0


def pl_likelihood(x, p=permutations, t=tau):
    N, M = p.shape
    restrict = t * 0.5 * npa.power(npa.sum(x), 2)
    result = 0.0
    for i in range(N):
        for j in range(M - 1):
            first = x[p[i][j]]
            # make log(sum(exp(x))) more stable:
            rem = logsumexp(x[p[i][j:]])
            result = result + first - rem
    return result - restrict


@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = npa.max(x)
    return max_x + npa.log(npa.sum(npa.exp(x - max_x)))


def neg_likelihood(x, p=permutations, t=tau):
    return -pl_likelihood(x, p, t)


def make_grad_logsumexp(ans, x):
    def gradient_product(g):
        return npa.full(x.shape, g) * npa.exp(x - npa.full(x.shape, ans))

    return gradient_product


logsumexp.defgrad(make_grad_logsumexp)


def wrapper(x):
    return neg_likelihood(x, p=permutations, t=tau)


def get_pl_parameters_for_rankings(rankings):
    def wrapper(x):
        return neg_likelihood(x, p=permutations, t=tau)

    permutations = rankings
    tau = 1.0
    x0 = npa.random.randn(rankings.shape[1])
    vg_like = value_and_grad(wrapper)
    hs_like = hvp(wrapper)
    result = minimize(vg_like, x0, jac=True, hessp=hs_like, method='Newton-CG')
    exped = npa.exp(result.x)
    normed = exped / exped.sum()
    return normed
