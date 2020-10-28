"""

Poisson matrix factorization with Batch inference and Stochastic inference

CREATED: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>

"""

import sys
import numpy as np
from scipy import special
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, TransformerMixin


class NetworkPoissonMF(BaseEstimator, TransformerMixin):
    ''' Poisson matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, tol=0.0005,
                 smoothness=100, random_state=None, verbose=False,
                 initialize_smart=False,
                 **kwargs):
        ''' Poisson matrix factorization

        Arguments
        ---------
        n_components : int
            Number of latent components

        max_iter : int
            Maximal number of iterations to perform

        tol : float
            The threshold on the increase of the objective to stop the
            iteration

        smoothness : int
            Smoothness on the initialization variational parameters

        random_state : int or RandomState
            Pseudo random number generator used for sampling

        verbose : bool
            Whether to show progress during model fitting

        **kwargs: dict
            Model hyperparameters
        '''

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose
        self.smart_init = initialize_smart

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        else:
            np.random.seed(0)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.5))
        self.b = float(kwargs.get('b', 0.5))
        self.c = float(kwargs.get('c', 1.))
        self.d = float(kwargs.get('d', 50.))

    def _nmf_initialize(self, A):
        nmf = NMF(n_components=self.n_components)
        z = nmf.fit_transform(A)
        z[z==0]=1e-10
        return z, np.log(z)
        # return np.exp(z), z

    def _init_weights(self, n_samples, A=None):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))

        if self.smart_init:
            self.Et, self.Elogt = self._nmf_initialize(A)
        else:
            self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

        self.c = 1. / np.mean(self.Et)

    def _init_non_identity_mat(self, N):
        self.non_id_mat = 1 - np.identity(N)

    def fit(self, A):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        
        n_samples, n_feats = A.shape
        n_users = A.shape[0]
        
        if self.smart_init:
            self._init_weights(n_samples, A=A)
        else:
            self._init_weights(n_samples)

        self._init_non_identity_mat(n_users)
                
        self._update(A)
        return self

    def transform(self, X, attr=None):
        '''Encode the data as a linear combination of the latent components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)

        attr: string
            The name of attribute, default 'Eb'. Can be changed to Elogb to
            obtain E_q[log beta] as transformed data.

        Returns
        -------
        X_new : array-like, shape(n_samples, n_filters)
            Transformed data, as specified by attr.
        '''

        if not hasattr(self, 'Eb'):
            raise ValueError('There are no pre-trained components.')
        n_samples, n_feats = X.shape
        if n_feats != self.Eb.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        if attr is None:
            attr = 'Et'
        self._init_weights(n_samples)
        self._update(X, update_beta=False)
        return getattr(self, attr)

    def _update(self, A):
        # alternating between update latent components and weights
        old_bd = -np.inf
        for i in range(self.max_iter):
            self._update_theta(A)
            bound = self._bound(A)
            if i>0:
                improvement = (bound - old_bd) / abs(old_bd)
                if self.verbose:
                    sys.stdout.write('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                     'Old objective: %.2f\t'
                                     'Improvement: %.5f' % (i, bound, old_bd,
                                                            improvement))
                    sys.stdout.flush()
                if improvement < self.tol:
                    break
            old_bd = bound
        if self.verbose:
            sys.stdout.write('\n')
        pass

    def _update_theta(self, A):
        ratio_adj = A / self._xexplog_adj()

        adj_term = np.multiply(np.exp(self.Elogt), np.dot(
            ratio_adj, np.exp(self.Elogt)))

        self.gamma_t = self.a + adj_term
        self.rho_t = np.dot(self.non_id_mat, self.Et)
        self.rho_t += self.c
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

    def _xexplog_adj(self):
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogt.T))


    def _bound(self, A):
        bound = np.sum(np.multiply(A, np.log(self._xexplog_adj()) - self.Et.dot(self.Et.T)))
        bound += _gamma_term(self.a, self.a * self.c,
                             self.gamma_t, self.rho_t,
                             self.Et, self.Elogt)
        bound += self.n_components * A.shape[0] * self.a * np.log(self.c)
        return bound


def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''    
    return (alpha / beta , special.psi(alpha) - np.log(beta))


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum(np.multiply((a - shape), Elogx) - np.multiply((b - rate), Ex) +
                  (special.gammaln(shape) - np.multiply(shape, np.log(rate))))

if __name__ == '__main__':
    N = 1000
    K = 20
    M = 1000

    Z = stats.gamma.rvs(0.5, scale=0.1, size=(N,K))
    A = stats.poisson.rvs(Z.dot(Z.T))
    A = np.triu(A)
    non_id = 1 - np.identity(N)
    A = A*non_id
    

    pmf = NetworkPoissonMF(n_components=K, verbose=True, initialize_smart=True)
    pmf.fit(A)
    print("MSE Z:", mse(Z, pmf.Et))