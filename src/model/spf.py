import sys
import numpy as np
from scipy import special
from scipy import sparse
import argparse
from scipy.stats import truncnorm, poisson, gamma
from sklearn.metrics import mean_squared_error as mse


class SocialPoissonFactorization:
	def __init__(self, n_components=100, max_iter=100, tol=0.0005, random_state=None, verbose=False, **kwargs):
		self.n_components = n_components
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self.verbose = verbose

		if type(self.random_state) is int:
			np.random.seed(self.random_state)
		elif self.random_state is not None:
			np.random.setstate(self.random_state)
		else:
			np.random.seed(0)

		self._parse_args()

	def _parse_args(self, **kwargs):
		self.learning_rate = float(kwargs.get('learning_rate', 0.1))
		self.batch_size = int(kwargs.get('batch_size', 100))

		self.inf_rate = float(kwargs.get('a', 0.1))
		self.inf_shp = float(kwargs.get('b', 0.1))

		self.item_mean = float(kwargs.get('c', 0.01))
		self.item_rate = float(kwargs.get('d', 10.))
		self.item_shp = self.item_mean*self.item_rate

		self.user_mean = float(kwargs.get('e', 0.01))
		self.user_rate = float(kwargs.get('f', 10.))
		self.user_shp = self.user_mean*self.user_rate

	def _init_gamma(self, M, K):
		self.gamma_shape = self.item_shp + truncnorm.rvs(0,1,size=(M,K))
		self.gamma_rates = self.item_rate + truncnorm.rvs(0,1,size=(M,K))
		self.E_log_gamma, self.E_gamma = self._compute_expectations(self.gamma_shape, self.gamma_rates)

	def _init_alpha(self, N, K):
		self.alpha_shape = self.user_shp + truncnorm.rvs(0, 1, size=(N,K))
		self.alpha_rates = self.user_rate + truncnorm.rvs(0, 1, size=(N,K))
		self.E_log_alpha, self.E_alpha = self._compute_expectations(self.alpha_shape, self.alpha_rates)

	def _init_beta(self, A, Y_past):
		N = A.shape[0]
		self.beta_shape = self.inf_shp + truncnorm.rvs(0,1,size=(N))
		self.beta_rates = self.inf_rate + A.sum(axis=0)*Y_past.sum(axis=1)
		self.E_log_beta, self.E_beta = self._compute_expectations(self.beta_shape, self.beta_rates)
		
	def _compute_expectations(self, shp, rte):
		return special.psi(shp) - np.log(rte), shp/rte
		
	def _compute_terms_and_normalizers(self, A, Y_past):
		self.beta_term = np.exp(self.E_log_beta)
		self.gamma_term = np.exp(self.E_log_gamma)
		self.alpha_term = np.exp(self.E_log_alpha)
		influence_component = (self.beta_term*A).dot(Y_past)
		preference_component = self.alpha_term.dot(self.gamma_term.T)
		self.normalizer = preference_component + influence_component

	##for now, this will simply return log likelihood under the Poisson model for Y
	def _compute_elbo(self, Y, A, Y_past):
		influence_rate = (self.E_beta*A).dot(Y_past)	
		pref_rate = self.E_alpha.dot(self.E_gamma.T)
		rate =  influence_rate + pref_rate
		return poisson.logpmf(Y, rate).sum()

	def _update_gamma(self, Y):
		normalized_obs = Y/self.normalizer
		expected_aux = self.gamma_term * normalized_obs.T.dot(self.alpha_term)
		self.gamma_shape = self.item_shp + expected_aux
		self.gamma_rates = self.item_rate + self.E_alpha.sum(axis=0, keepdims=True)
		self.E_log_gamma, self.E_gamma = self._compute_expectations(self.gamma_shape, self.gamma_rates)

	def _update_alpha(self, Y):
		normalized_obs = Y/self.normalizer
		expected_aux = self.alpha_term * normalized_obs.dot(self.gamma_term)
		self.alpha_shape = self.user_shp + expected_aux
		self.alpha_rates = self.user_shp + self.E_gamma.sum(axis=0, keepdims=True)
		self.E_log_alpha, self.E_alpha = self._compute_expectations(self.alpha_shape, self.alpha_rates)

	def _update_beta(self, Y, Y_past, A):
		normalized_obs = Y/self.normalizer
		expected_aux = self.beta_term * (A * normalized_obs.dot(Y_past.T)).sum(axis=0)
		self.beta_shape = self.inf_shp + expected_aux
		self.E_log_beta, self.E_beta = self._compute_expectations(self.beta_shape, self.beta_rates)


	def fit(self, Y, A, Y_past):

		N = Y.shape[0]
		M = Y.shape[1]
		K = self.n_components

		self._init_beta(A, Y_past)
		self._init_gamma(M,K)
		self._init_alpha(N,K)	
		
		old_bd = float('-inf')
		bd = self._compute_elbo(Y, A, Y_past)

		for i in range(self.max_iter):
			if self.verbose:
				print("Bound:", bd)
				sys.stdout.flush()

			old_bd = bd

			self._compute_terms_and_normalizers(A, Y_past)
			self._update_beta(Y, Y_past, A)
			self._update_gamma(Y)
			self._update_alpha(Y)
			bd = self._compute_elbo(Y, A, Y_past)

			if (bd-old_bd)/abs(old_bd) < self.tol:
				print(old_bd, bd)
				break

def get_set_overlap(Beta_p, Beta, k=20):
    top = np.argsort(Beta)[-k:]
    top_p = np.argsort(Beta_p)[-k:]
    return np.intersect1d(top, top_p).shape[0]/k

if __name__ == '__main__':
	N = 1000
	K = 10
	M = 1000

	Gamma = gamma.rvs(0.5, scale=0.1, size=(M,K))
	Alpha = gamma.rvs(0.5, scale=0.1, size=(N,K))
	Beta = gamma.rvs(0.005, scale=10., size=N)

	A = poisson.rvs(Alpha.dot(Alpha.T))
	non_id = 1 - np.identity(N)
	A = A*non_id
	
	rate_pref = Alpha.dot(Gamma.T)
	Y_past = poisson.rvs(rate_pref)
	rate_inf = (Beta*A).dot(Y_past)
	Y = poisson.rvs(rate_pref + rate_inf)
	print("Sparsity of data matrices:", A.mean(), Y_past.mean(), Y.mean())

	pmf = SocialPoissonFactorization(n_components=K, verbose=True)
	pmf.fit(Y, A, Y_past)

	print("Beta overlap:", get_set_overlap(pmf.E_beta, Beta))
	print("MSE Beta:", mse(Beta, pmf.E_beta))
	print("MSE Random Beta:", mse(Beta, gamma.rvs(0.1, scale=10., size=N)))
