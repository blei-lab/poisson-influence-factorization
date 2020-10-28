from pokec.process_dataset import process_dataset_multi_covariate, make_multi_covariate_simulation

import model.multi_cause_influence as causal
import model.network_model as nm
import model.pmf as pmf

import numpy as np
import os
import argparse
import sys
from sklearn.metrics import mean_squared_error as mse, roc_auc_score
from sklearn.decomposition import NMF
from itertools import product
from scipy.stats import poisson

def calculate_ppc(items, array, z, w):
	users = np.arange(array.shape[0])
	rates = z.dot(w.T)[users, items]
	replicated = poisson.rvs(rates)
	heldout = array[users,items]
	logll_heldout = poisson.logpmf(heldout, rates).sum()
	logll_replicated = poisson.logpmf(replicated, rates).sum()
	return logll_heldout, logll_replicated

def evaluate_random_subset(items, array, z, w, metric='logll'):
	users = np.arange(array.shape[0])
	expected = z.dot(w.T)[users,items]
	truth = array[users,items]
	if metric == 'auc':
		return roc_auc_score(truth,expected)
	else:
		return poisson.logpmf(truth, expected).sum()

def mask_items(N,M):
	items = np.arange(M)
	random_items = np.random.choice(items, size=N)
	return random_items

def main():
	num_exps = 20
	Ks = [3,5,8,10]
	mixture_pr = 0.5
	noise = 10.
	conf_strength = 50.
	
	a_score = np.zeros((num_exps, len(Ks)))
	x_score = np.zeros((num_exps, len(Ks)))
	x_auc = np.zeros((num_exps, len(Ks)))
	for e_idx in range(num_exps):
		print("Working on experiment", e_idx)

		A, users, user_one_hots, item_one_hots, Beta = process_dataset_multi_covariate(datapath=datadir, sample_size=3000, num_items=3000, influence_shp=0.005, covar_2='random', covar_2_num_cats=5,use_fixed_graph=False)
		Y, Y_past, Z, Gamma, Alpha, W = make_multi_covariate_simulation(A, user_one_hots, item_one_hots, 
				Beta, 
				noise=noise,
				confounding_strength=conf_strength,
				mixture_prob=mixture_pr)

		N = Y_past.shape[0]
		M = Y_past.shape[1]
		masked_friends = mask_items(N, N)
		past_masked_items = mask_items(N,M)

		Y_past_train = Y_past.copy()
		A_train = A.copy()

		users = np.arange(N)
		A_train[users, masked_friends] = 0
		Y_past_train[users, past_masked_items] = 0

		for k_idx, K in enumerate(Ks):
			network_model = nm.NetworkPoissonMF(n_components=K, verbose=False)
			network_model.fit(A_train)
			Z_hat = network_model.Et
			pmf_model = pmf.PoissonMF(n_components=K, verbose=False)
			pmf_model.fit(Y_past_train)
			W_hat = pmf_model.Eb.T
			Theta_hat = pmf_model.Et

			replicates=100
			A_predictive_score = 0.0
			YP_pred_score = 0.0
			for _ in range(replicates):
				A_logll_heldout, A_logll_replicated = calculate_ppc(masked_friends, A, Z_hat, Z_hat)
				Y_logll_heldout, Y_logll_replicated = calculate_ppc(past_masked_items, Y_past, Theta_hat, W_hat)
				if A_logll_replicated > A_logll_heldout:
					A_predictive_score += 1.
				if Y_logll_replicated > Y_logll_heldout:
					YP_pred_score += 1.

			a_score[e_idx][k_idx] = A_predictive_score/replicates
			x_score[e_idx][k_idx] = YP_pred_score/replicates
			x_auc[e_idx][k_idx] = evaluate_random_subset(past_masked_items, Y_past, Theta_hat, W_hat,metric='logll')

	print("A ppc scores across choices of num components:", a_score.mean(axis=0))
	print("X ppc scores across choices of num components:", x_score.mean(axis=0))
	print("X auc across choices of num components:", x_auc.mean(axis=0))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-dir", action="store", default='../dat/pokec/regional_subset')
	args = parser.parse_args()
	datadir = args.data_dir

	main()