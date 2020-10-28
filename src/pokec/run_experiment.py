from pokec.process_dataset import PokecSimulator

import model.spf as spf
import model.network_model as nm
import model.pmf as pmf
import model.multi_cause_influence as causal
import model.joint_factor_model as joint

import numpy as np
import os
import argparse
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import NMF
from itertools import product

def post_process_influence(X, Beta):
	total_X = X.sum(axis=1)
	no_X = total_X == 0
	Beta[no_X] = 1.
	return Beta

def get_average_embeddings(A, E, average=True):
	embed = A.dot(E)
	if average:
		N_friends = A.sum(axis=1)
		has_friends = N_friends > 0
		embed[has_friends,:] = embed[has_friends,:]/N_friends[has_friends][:,np.newaxis]
	return embed

def get_set_overlap(Beta_p, Beta, k=50):
	top = np.argsort(Beta)[-k:]
	top_p = np.argsort(Beta_p)[-k:]
	return np.intersect1d(top, top_p).shape[0]/np.union1d(top, top_p).shape[0]

def main():
	write = os.path.join(outdir, model + '.' + variant + '_model_fitted_params')
	os.makedirs(write, exist_ok=True)

	simulation_model = PokecSimulator(datapath=datadir, subnetwork_size=3000, num_items=3000, influence_shp=0.005, covar_2='random', covar_2_num_cats=5, seed=seed)
	simulation_model.process_dataset()
	A = simulation_model.A
	print("Adj. size and mean:", A.shape, A.mean())

	for ct in confounding_type:
		for (noise, confounding) in confounding_configs:
			print("Working on confounding setting with prob:", ct, "and cov. 1/cov. 2 confounding strength:", (noise, confounding))
			sys.stdout.flush()

			Y, Y_past = simulation_model.make_multi_covariate_simulation(noise=noise, 
				confounding_strength=confounding, 
				confounding_to_use=ct)

			Beta = simulation_model.beta
			Z = simulation_model.user_embed_1
			Gamma = simulation_model.item_embed_1
			Alpha = simulation_model.user_embed_2
			W = simulation_model.item_embed_2

			Beta = post_process_influence(Y_past, Beta)

			N = Y_past.shape[0]
			M = Y_past.shape[1]

			if model == 'unadjusted':
				m = causal.CausalInfluenceModel(n_components=K, n_exog_components=P, verbose=True, model_mode='influence_only')

			elif model == 'network_pref_only':
				m = causal.CausalInfluenceModel(n_components=K, n_exog_components=P, verbose=True, model_mode='network_preferences')

			elif model == 'item_only':
				m = causal.CausalInfluenceModel(n_components=K, n_exog_components=P, verbose=True, model_mode='item')

			elif model == 'pif':
				m = causal.CausalInfluenceModel(n_components=K+P, n_exog_components=P, verbose=True, model_mode='full')

			elif model == 'spf':
				m = spf.SocialPoissonFactorization(n_components=K+P, verbose=True)

			elif model == 'no_unobs':
				num_regions = Z.shape[1]
				num_covar_comps = W.shape[1]
				m = causal.CausalInfluenceModel(n_components=num_regions, n_exog_components=num_covar_comps, verbose=True, model_mode='full')

			elif model == 'item_only_oracle':
				num_regions = Z.shape[1]
				num_covar_comps = W.shape[1]
				m = causal.CausalInfluenceModel(n_components=num_regions, n_exog_components=num_covar_comps, verbose=True, model_mode='item')

			elif model == 'network_only_oracle':
				num_regions = Z.shape[1]
				num_covar_comps = W.shape[1]
				m = causal.CausalInfluenceModel(n_components=num_regions, n_exog_components=num_covar_comps, verbose=True, model_mode='network_preferences')

			if model == 'spf':
				m.fit(Y, A, Y_past)
			elif model == 'no_unobs':
				m.fit(Y, A, Z, W, Y_past)
			elif model == 'item_only_oracle':
				m.fit(Y, A, Z, W, Y_past)
			elif model == 'network_only_oracle':
				m.fit(Y, A, Z, W, Y_past)
			elif model == 'network_pref_only':
				network_model = nm.NetworkPoissonMF(n_components=K)
				network_model.fit(A)
				Z_hat = network_model.Et
				W_hat = np.zeros((M,P))
				m.fit(Y, A, Z_hat, W_hat, Y_past)
			elif model == 'item_only':
				pmf_model = pmf.PoissonMF(n_components=P)
				pmf_model.fit(Y_past)
				W_hat = pmf_model.Eb.T
				Z_hat = np.zeros((N,K))
				m.fit(Y, A, Z_hat, W_hat, Y_past)
			elif model=='unadjusted':
				Z_hat = np.zeros((N,K))
				W_hat = np.zeros((M,P))
				m.fit(Y, A, Z_hat, W_hat, Y_past)
			else:
				if variant == 'z-theta-joint':
					joint_model = joint.JointPoissonMF(n_components=K)
					joint_model.fit(Y_past, A)
					Z_hat_joint = joint_model.Et
					# W_hat = joint_model.Eb.T

					pmf_model = pmf.PoissonMF(n_components=P)
					pmf_model.fit(Y_past)
					W_hat = pmf_model.Eb.T

				elif variant =='theta-only':
					pmf_model = pmf.PoissonMF(n_components=P)
					pmf_model.fit(Y_past)
					W_hat = pmf_model.Eb.T
					Theta_hat = pmf_model.Et

				elif variant == 'z-theta-concat':
					network_model = nm.NetworkPoissonMF(n_components=K)
					network_model.fit(A)
					Z_hat = network_model.Et

					pmf_model = pmf.PoissonMF(n_components=P)
					pmf_model.fit(Y_past)
					W_hat = pmf_model.Eb.T
					Theta_hat = pmf_model.Et
				else:
					network_model = nm.NetworkPoissonMF(n_components=K)
					network_model.fit(A)
					Z_hat = network_model.Et

					pmf_model = pmf.PoissonMF(n_components=P)
					pmf_model.fit(Y_past)
					W_hat = pmf_model.Eb.T
					Theta_hat = pmf_model.Et

				Rho_hat = np.zeros((N, K+P))
				if variant=='z-only':
					Rho_hat[:,:K] = Z_hat
				elif variant=='theta-only':
					Rho_hat[:,:P] = Theta_hat
				elif variant=='z-theta-concat':
					Rho_hat = np.column_stack((Z_hat, Theta_hat))
				else:
					Rho_hat[:,:K] = Z_hat_joint

				m.fit(Y, A, Rho_hat, W_hat, Y_past)
				
			Beta_p = m.E_beta
			score = get_set_overlap(Beta_p, Beta)
			loss = mse(Beta, Beta_p)

			print("Overlap:", score, "MSE:", loss)
			print("*"*60)
			sys.stdout.flush()
			outfile = os.path.join(write, 'conf=' + str((noise, confounding)) + ';conf_type=' + ct)
			np.savez_compressed(outfile, fitted=Beta_p, true=Beta)
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-dir", action="store", default='../dat/pokec/regional_subset')
	parser.add_argument("--out-dir", action="store", default="../out/")
	parser.add_argument("--model", action="store", default='pif')
	parser.add_argument("--variant", action="store", default='z-theta-joint')
	parser.add_argument("--confounding_type", action="store", default="both")
	parser.add_argument("--configs", action="store", default='50,50')
	parser.add_argument("--num_components", action="store", default=10, type=int)
	parser.add_argument("--num_exog_components", action="store", default=10, type=int)
	parser.add_argument("--seed", action="store", default=10, type=int)

	args = parser.parse_args()
	datadir = args.data_dir
	outdir = args.out_dir
	model = args.model
	variant = args.variant

	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	confounding_type = args.confounding_type
	configs = args.configs
	K = args.num_components
	P = args.num_exog_components
	seed = args.seed

	confounding_type = confounding_type.split(',')
	confounding_configs = [(int(c.split(',')[0]), int(c.split(',')[1])) for c in configs.split(':')]

	print("Confounding configs:", confounding_configs)
	print("Model:", model)

	main()