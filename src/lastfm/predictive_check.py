from pokec.process_dataset import process_dataset_multi_covariate, make_multi_covariate_simulation

import model.multi_cause_influence as causal
import model.network_model as nm
import model.pmf as pmf
import model.joint_factor_model as joint

import numpy as np
import os
import argparse
import sys
from sklearn.metrics import mean_squared_error as mse, roc_auc_score
from sklearn.decomposition import NMF
from itertools import product
from scipy.stats import poisson, gamma

def load_data(file='../dat/lastfm/lastfm_processed.npz'):
	array = np.load(file)
	A = array['adj']
	Y_past = array['y_past']
	Y = array['y']
	return A, Y_past, Y

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

def evaluate_influence_model(items, y, y_p, a, z, w, inf):
	users = np.arange(y.shape[0])
	inf_rate = (inf*a).dot(y_p)
	expected = (z.dot(w.T) + inf_rate)[users,items]
	truth = y[users,items]
	return poisson.logpmf(truth, expected).mean()

def evaluate_multi_view_influence(items, y, y_p, a, z, w, alpha, gamma, inf, theta=None, kappa=None):
	users = np.arange(y.shape[0])
	inf_rate = (inf*a).dot(y_p)
	rate = z.dot(gamma.T) + alpha.dot(w.T) 
	if theta is not None:
		rate += theta.dot(kappa.T)
	expected = (rate + inf_rate)[users,items]
	truth = y[users,items]
	return poisson.logpmf(truth, expected).mean()

def mask_items(N,M):
	items = np.arange(M)
	random_items = np.random.choice(items, size=N)
	return random_items

def main():
	# os.makedirs(outdir, exist_ok=True)
	A, Y_past, Y = load_data()
	num_exps = 5
	Ks = [20, 30, 50, 80]
	
	a_score = np.zeros((num_exps, len(Ks)))
	x_score = np.zeros((num_exps, len(Ks)))
	x_random = np.zeros((num_exps, len(Ks)))
	for e_idx in range(num_exps):
		print("Working on experiment", e_idx)

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
			network_model = joint.JointPoissonMF(n_components=K)
			network_model.fit(Y_past_train, A_train)
			Z_hat = network_model.Et
			pmf_model = pmf.PoissonMF(n_components=K)
			pmf_model.fit(Y_past_train)
			W_hat = pmf_model.Eb.T
			Theta_hat = pmf_model.Et

			a_score[e_idx][k_idx] = evaluate_random_subset(past_masked_items, Y_past, Theta_hat, W_hat,metric='auc')
			x_score[e_idx][k_idx] = evaluate_random_subset(past_masked_items, Y_past, Theta_hat, W_hat,metric='logll')

			smoothness = 100.
			Theta_random = smoothness \
							* np.random.gamma(smoothness, 1. / smoothness, size=Theta_hat.shape)
			W_random = smoothness \
							* np.random.gamma(smoothness, 1. / smoothness, size=W_hat.shape)

							
			x_random[e_idx][k_idx] = evaluate_random_subset(past_masked_items, Y_past, Theta_random, W_random, metric='logll')

			# replicates=10
			# A_predictive_score = 0.0
			# YP_pred_score = 0.0
			# for _ in range(replicates):
			# 	A_logll_heldout, A_logll_replicated = calculate_ppc(masked_friends, A, Z_hat, Z_hat)
			# 	Y_logll_heldout, Y_logll_replicated = calculate_ppc(past_masked_items, Y_past, Theta_hat, W_hat)
			# 	if A_logll_replicated > A_logll_heldout:
			# 		A_predictive_score += 1.
			# 	if Y_logll_replicated > Y_logll_heldout:
			# 		YP_pred_score += 1.

			# a_score[e_idx][k_idx] = A_predictive_score/replicates
			# x_score[e_idx][k_idx] = YP_pred_score/replicates
			# x_auc[e_idx][k_idx] = evaluate_random_subset(past_masked_items, Y_past, Theta_hat, W_hat,metric='logll')

	print("A scores:", a_score.mean(axis=0))
	print("X scores:", x_score.mean(axis=0))
	print("X random scores:", x_random.mean(axis=0))

if __name__ == '__main__':
	main()