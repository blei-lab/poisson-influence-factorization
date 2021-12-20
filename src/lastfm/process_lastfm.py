from lastfm import lastfm_utils as ut
import numpy as np
from collections import Counter
import os

def main():
	friend_data = ut.extract_friends()

	### find all people who have both in-edges and out-edges
	users = set(friend_data.keys())
	friends = set().union(*friend_data.values())
	intersection = users & friends

	## count up in and out edges
	in_edges = Counter()
	out_edges = Counter()
	for u in intersection:
		for f in friend_data[u]:
			if f in intersection:
				in_edges.update([f])
				out_edges.update([u])

	## select the people who are among top 5k in both in and out edges
	most_out_edges = [t[0] for t in out_edges.most_common(10000)]
	most_in_edges = [t[0] for t in in_edges.most_common(10000)]
	selected_users = set(most_in_edges) & set(most_out_edges)

	# make adj matrix
	adj_matrix, uid_map, _ = ut.make_matrix(friend_data, selected_users, restrict_items=True)
	print("Adjacency matrix shape and mean:", adj_matrix.shape, adj_matrix.mean())

	## extract all activities for selected users
	activity_dict = ut.get_activities(selected_users)

	## find the most popular activities (in terms of num users who participated)
	activity_counts = Counter()
	for u, activity_list in activity_dict.items():
		for activity in activity_list:
			activity_counts.update([activity[0]])
	most_popular = set(t[0] for t in activity_counts.most_common(6000))

	## get all timestamps that correspond to those activities
	valid_timestamps = [activity[1] for activity_list in activity_dict.values() for activity in activity_list if activity[0] in most_popular]

	## select the first quartile of timestamps and get all user-activity interactions within that time period as y past
	quantile=0.25
	threshold = np.quantile(valid_timestamps, quantile)
	past_obs = {u:[a[0] for a in act if a[1] <= threshold and a[0] in most_popular] for u, act in activity_dict.items()}
	valid_acts = set([a for u, activities in past_obs.items() for a in activities])
	Y_past, _, _ = ut.make_matrix(past_obs, selected_users)
	print("Past obs. matrix is of size:", Y_past.shape)
	print(Y_past.mean())

	## select middle quartile and get the user-activity links for selected users and selected activities as y
	median=0.6
	upper_cutoff = np.quantile(valid_timestamps, median)
	present_obs = {u:[a[0] for a in act if a[1] > threshold and a[1] <= upper_cutoff and a[0] in valid_acts] for u, act in activity_dict.items()}
	Y, _, _ = ut.make_matrix(present_obs, selected_users)
	print("Present obs. matrix is of size:", Y.shape)
	print(Y.mean())

	has_past_purchases = (Y_past.sum(axis=1) > 1)
	has_present_purchases = (Y.sum(axis=1) > 1)
	valid = has_past_purchases & has_present_purchases
	Y_past = Y_past[valid, :]
	Y = Y[valid,:]

	adj_matrix = adj_matrix[valid,:]
	adj_matrix = adj_matrix[:,valid]
	print("Adjacency matrix shape and mean:", adj_matrix.shape, adj_matrix.mean())

	print("Past obs. matrix is of size:", Y_past.shape)
	print(Y_past.mean())

	print("Present obs. matrix is of size:", Y.shape)
	print(Y.mean())

	future=0.99
	final_cutoff = np.quantile(valid_timestamps, future)
	heldout_obs = {u:[a[0] for a in act if a[1] > upper_cutoff and a[1] <= final_cutoff and a[0] in valid_acts] for u, act in activity_dict.items()}
	Y_heldout, _, _ = ut.make_matrix(heldout_obs, selected_users)
	Y_heldout = Y_heldout[valid, :]
	print("Future heldout obs. matrix is of size:", Y_heldout.shape)
	print(Y_heldout.mean())


	# write = '../dat/lastfm/'
	# os.makedirs(write, exist_ok=True)
	# np.savez_compressed(write + 'lastfm_processed', adj=adj_matrix, y_past=Y_past, y=Y, y_heldout=Y_heldout)

if __name__ == '__main__':
	main()

