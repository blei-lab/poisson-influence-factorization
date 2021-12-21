import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pickle
from scipy.stats import gamma, poisson, bernoulli
from scipy.special import expit
import sys

class PokecSimulator():
	def __init__(self, datapath='../dat/pokec/regional_subset', subnetwork_size=3000, influence_shp=0.005, num_items=3000, covar_1='region_categorical', covar_2='random', covar_2_num_cats=5, **kwargs):
		self.datapath = datapath
		self.subnetwork_size = subnetwork_size
		self.influence_shp = influence_shp
		self.num_items = num_items
		self.covar_1 = covar_1
		self.covar_2 = covar_2
		self.covar_2_num_cats = covar_2_num_cats

		self.parse_args(**kwargs)

	def parse_args(self, **kwargs):
		self.random_seed = int(kwargs.get('seed', 12345))
		self.do_sensitivity = bool(kwargs.get('do_sensitivity', False))
		self.sensitivity_parameter = float(kwargs.get('sensitivity_parameter', 1.))
		self.error_rate = float(kwargs.get('error_rate', 0.3))
		np.random.seed(self.random_seed)

	def snowball_sample(self):
		sampled_users = set()
		users = np.arange(self.A.shape[0])
		np.random.shuffle(users)
		u_iter = 0
		while(len(sampled_users) < self.subnetwork_size):
			user = users[u_iter]
			friends = np.nonzero(self.A[u_iter, :])[0]
			sampled_users.add(user)
			sampled_users |= set(list(friends))
			if friends.shape[0] > 2:
				u_iter = np.random.choice(friends)
			else:
				u_iter +=1
		return np.array(list(sampled_users))

	def load_edgelist(self):
		arr = np.load(os.path.join(self.datapath, 'pokec_links.npz'))
		self.edgelist = arr['edge_list']
		
	def load_profiles(self):
		with open(os.path.join(self.datapath, 'profiles.pkl'), 'rb') as f:
			self.profiles = pickle.load(f)
		self.profiles['region_categorical'] = pd.Categorical(self.profiles.region).codes
		code = {r:i for (i,r) in enumerate(np.unique(self.profiles['region_categorical']))}
		self.profiles['region_categorical'] = self.profiles['region_categorical'].apply(lambda x: code[x])
		age_bins = np.arange(0,100,step=10)
		self.profiles['age_binned'] = np.digitize(self.profiles.age.values, age_bins)
		self.profiles['age_binned']-=1
		self.uids = self.profiles.index.values
		
	def make_adj_matrix(self):
		row_inds = self.edgelist[:,0]
		col_inds = self.edgelist[:,1]
		data = np.ones(row_inds.shape[0])
		row_inds.shape, col_inds.shape, data.shape
		A = csr_matrix((data, (row_inds, col_inds)))
		self.A = A.toarray()

	def get_one_hot_covariate_encoding(self, covar):
		categories = np.unique(self.profiles[covar])
		num_cats = len(categories)
		one_hot_encoding = np.zeros((self.users.shape[0], num_cats))
		data = self.profiles.loc[self.uids[self.users], covar].values
		u_idx = np.arange(self.users.shape[0])
		one_hot_encoding[u_idx,data] = 1
		return one_hot_encoding

	def sample_random_covariate(self, num_categories, num_samples):
		one_hot_encoding = np.zeros((num_samples,num_categories))
		one_hot_encoding[np.arange(num_samples),np.random.randint(0,num_categories,size=num_samples)] = 1
		return one_hot_encoding

	def make_embeddings(self, user_encoding, item_encoding, num_cats, noise=10.,confounding_strength=10., gamma_mean=0.1, gamma_scale=0.1):
		M = item_encoding.shape[0]
		N = user_encoding.shape[0]
		gamma_shp = gamma_mean/gamma_scale
		embedding_shp = (num_cats*gamma_shp*noise)/(noise + (num_cats-1))
		loadings_shp = (num_cats*gamma_shp*confounding_strength)/(confounding_strength + (num_cats-1))

		embedding = user_encoding*gamma.rvs(embedding_shp, scale=gamma_scale, size=(N, num_cats))
		embedding += (1-user_encoding)*gamma.rvs(embedding_shp/noise, scale=gamma_scale, size=(N, num_cats))

		loadings = item_encoding*gamma.rvs(loadings_shp, scale=gamma_scale, size=(M, num_cats))
		loadings += (1-item_encoding)*gamma.rvs(loadings_shp/confounding_strength, scale=gamma_scale, size=(M, num_cats))
		return embedding, loadings

	def make_simulated_influence(self):
		N = self.A.shape[0]
		if self.influence_shp > 0:
			influence = gamma.rvs(self.influence_shp, scale=10., size=N)
		else:
			influence = np.zeros(N)
		return influence

	def process_dataset(self):
		self.load_edgelist()
		self.load_profiles()
		self.make_adj_matrix()
		
		self.users = self.snowball_sample()
		self.A = self.A[self.users,:]
		self.A = self.A[:,self.users]
		
		self.user_one_hot_covar_1 = self.get_one_hot_covariate_encoding(self.covar_1)
		
		if self.covar_2=='age_binned':
			self.user_one_hot_covar_2 = self.get_one_hot_covariate_encoding(self.covar_2)
		else:
			self.user_one_hot_covar_2 = self.sample_random_covariate(self.covar_2_num_cats, self.A.shape[0])

		num_regions = self.user_one_hot_covar_1.shape[1]
		num_cats = self.covar_2_num_cats
		self.item_one_hot_covar_1 = self.sample_random_covariate(num_regions, self.num_items)
		self.item_one_hot_covar_2 = self.sample_random_covariate(num_cats, self.num_items)
		self.beta = self.make_simulated_influence()
		no_friends = self.A.sum(axis=0) == 0
		self.beta[no_friends] = 1.


	def make_multi_covariate_simulation(self, noise=10., confounding_strength=10.,gamma_mean=0.1, gamma_scale=0.1, confounding_to_use='both'):
		covar_1_num_cats = self.user_one_hot_covar_1.shape[1]
		covar_2_num_cats = self.covar_2_num_cats

		self.user_embed_1, self.item_embed_1 = self.make_embeddings(self.user_one_hot_covar_1, 
			self.item_one_hot_covar_1,
			covar_1_num_cats,
			noise=noise,
			confounding_strength=confounding_strength,
			gamma_mean=gamma_mean,
			gamma_scale=gamma_scale)

		self.user_embed_2, self.item_embed_2 = self.make_embeddings(self.user_one_hot_covar_2, 
			self.item_one_hot_covar_2,
			covar_2_num_cats,
			noise=noise,
			confounding_strength=confounding_strength,
			gamma_mean=gamma_mean,
			gamma_scale=gamma_scale)

		Y, Y_past = self.make_mixture_preferences_outcomes(confounding_to_use=confounding_to_use)
		return Y, Y_past

	def make_mixture_preferences_outcomes(self, confounding_to_use):
		homophily_pref = self.user_embed_1.dot(self.item_embed_1.T)
		random_pref = self.user_embed_2.dot(self.item_embed_2.T)

		if confounding_to_use == 'homophily':
			base_rate = homophily_pref
		elif confounding_to_use == 'both':
			base_rate = homophily_pref + random_pref
		else:
			base_rate = random_pref

	
		if self.do_sensitivity:
			bias = self.create_bias()
			y_past = poisson.rvs(base_rate + bias)
			influence_rate = (self.beta * self.A).dot(y_past)
			y = poisson.rvs(base_rate + influence_rate + bias)
		else:
			y_past = poisson.rvs(base_rate)
			influence_rate = (self.beta * self.A).dot(y_past)
			y = poisson.rvs(base_rate + influence_rate)
		
		return y, y_past

	def create_bias(self, gamma_mean=0.1, gamma_scale=0.1):
		N = self.A.shape[0]
		M = self.num_items

		bias = np.zeros((N,M))
		(u1, u2) = np.nonzero(self.A)
		mask = bernoulli.rvs(self.error_rate, size=u1.shape[0])
		
		for edge_iter in range(u1.shape[0]):
			if mask[edge_iter]:
				i = u1[edge_iter]
				j = u2[edge_iter]
				bias_mean = gamma_mean / self.sensitivity_parameter
				bias_val = gamma.rvs((bias_mean/gamma_scale), scale=gamma_scale)
				random_items = bernoulli.rvs(self.error_rate, size=self.num_items)
				for k in np.nonzero(random_items):
					bias[i,k] = bias_val
					bias[j,k] = bias_val

		return bias