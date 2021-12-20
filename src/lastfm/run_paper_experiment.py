import model.spf as spf
import model.network_model as nm
import model.pmf as pmf
import model.multi_cause_influence as causal
import model.joint_factor_model as joint

import numpy as np
import os
import argparse
import sys

def load_data(file='../dat/lastfm/lastfm_processed.npz'):
	array = np.load(file)
	A = array['adj']
	Y_past = array['y_past']
	Y = array['y']
	return A, Y_past, Y

def main():
	write = os.path.join(outdir, model + '_fitted_params')
	os.makedirs(write, exist_ok=True)
	
	A, Y_past, Y = load_data()

	N = Y_past.shape[0]
	M = Y_past.shape[1]
	K = 20
	P = 20

	Beta_hat = np.zeros(N)
	Gamma_hat = np.zeros((M,K))
	Alpha_hat = np.zeros((N,P))
	W_hat = np.zeros((M,P))
	Z_hat = np.zeros((N,K))

	if model == 'unadjusted':
		m = causal.CausalInfluenceModel(n_components=K, n_exog_components=P, verbose=True, model_mode='influence_only')

	elif model == 'network_pref_only':
		m = causal.CausalInfluenceModel(n_components=K, n_exog_components=P, verbose=True, model_mode='network_preferences')

	elif model == 'item_only':
		m = causal.CausalInfluenceModel(n_components=K, n_exog_components=P, verbose=True, model_mode='item')

	elif model == 'pif':
		m = causal.CausalInfluenceModel(n_components=K, n_exog_components=P, verbose=True, model_mode='full')

	elif model == 'spf':
		m = spf.SocialPoissonFactorization(n_components=K+P, verbose=True)

	if model == 'spf':
		m.fit(Y, A, Y_past)
	
	elif model == 'network_pref_only':
		network_model = nm.NetworkPoissonMF(n_components=K)
		network_model.fit(A)
		Z_hat = network_model.Et
		m.fit(Y, A, Z_hat, W_hat, Y_past)

	elif model == 'item_only':
		pmf_model = pmf.PoissonMF(n_components=P)
		pmf_model.fit(Y_past)
		W_hat = pmf_model.Eb.T
		m.fit(Y, A, Z_hat, W_hat, Y_past)

	elif model=='unadjusted':
		m.fit(Y, A, Z_hat, W_hat, Y_past)
	else:
		joint_model = joint.JointPoissonMF(n_components=K)
		joint_model.fit(Y_past, A)
		Z_hat_joint = joint_model.Et

		pmf_model = pmf.PoissonMF(n_components=P)
		pmf_model.fit(Y_past)
		W_hat = pmf_model.Eb.T

		m.fit(Y, A, Z_hat_joint, W_hat, Y_past)
		

	if model == 'pif' or model == 'spf' or model == 'network_pref_only':
		Gamma_hat = m.E_gamma

	if model == 'pif' or model == 'item_only':
		Alpha_hat = m.E_alpha

	if model == 'spf':
		Z_hat = m.E_alpha

	if model == 'pif':
		Z_hat = Z_hat_joint

	Beta_hat = m.E_beta
	np.savez_compressed(write, Z_hat=Z_hat, W_hat=W_hat, Alpha_hat=Alpha_hat, Gamma_hat=Gamma_hat, Beta_hat=Beta_hat)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--out-dir", action="store", default="../out/lastfm/")
	parser.add_argument("--model", action="store", default='pif')

	args = parser.parse_args()
	outdir = args.out_dir
	model = args.model

	print("Model:", model)

	main()