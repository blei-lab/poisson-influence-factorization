import os
import numpy as np
import pandas as pd
import dill

np.random.seed(12345)

datadir = os.path.join('..', 'dat', 'lastfm_filtered')


max_file = 430000
increments = 10000
file_idx = np.arange(0,max_file+increments, step=increments)

def make_matrix(user_item_dict, users, restrict_items=False):

	uids = {u:i for i,u in enumerate(users)}
	unique_items = set().union(*user_item_dict.values())
	num_items = len(unique_items)
	num_users = len(uids)
	it_ids = {it:i for i, it in enumerate(list(unique_items))}

	if restrict_items:
		num_items = num_users

	data = np.zeros((num_users, num_items))

	for u in users:
		if u not in user_item_dict.keys():
			continue

		items = user_item_dict[u]
		uid = uids[u]
		# it_id = [it_ids[it] for it in items]
		for it in items:
			if restrict_items:
				if it in uids.keys():
					j = uids[it]
					data[uid,j] += 1
			else:
				j = it_ids[it]
				data[uid, j] += 1

	return data, uids, it_ids


def load_friends(datapath=datadir):
	outdir = os.path.join(datapath, 'dicts')
	with open(os.path.join(outdir, 'friends.dat'), 'rb') as f:
		friend_data = dill.load(f)
	return friend_data


def snowball_sample(friend_data, size=5000):
	users = set()
	user_list = list(friend_data.keys())
	while len(users) < size:
		u = np.random.choice(user_list)
		if u not in friend_data.keys():
			continue

		# u = np.random.choice(user_list)
		
		users |= {u} | set(friend_data[u])

		user_list = friend_data[u]

	return users


def extract_friends(datapath=datadir):
	friend_data = {}

	for f_idx in file_idx:
		with open(os.path.join(datapath, 'egodata_' + str(f_idx) + '.tsv'), 'r') as f:
			lines = f.readlines()
			for idx, line in enumerate(lines):
				if "#" == line[0]:
					continue
				line = line.strip()
				
				if (idx-1) % 5 == 0:
					line = line.split(' ')
					userid = int(line[0])
					hasfriends = int(line[1])

					if hasfriends:
						if 'None' in lines[idx+1]:continue
						friends = [int(f) for f in lines[idx+1].strip().split(' ')]
						friend_data[userid] = friends
						
	# outdir = os.path.join(datapath, 'dicts')
	# os.makedirs(outdir, exist_ok=True)
	# with open(os.path.join(outdir, 'friends.dat'), 'wb') as f:
	# 	dill.dump(friend_data, f)
	return friend_data

def get_activities(userid_list, datapath=datadir):

	num_files = max_file // increments
	offset=4
	activity_data = {}

	for i in range(num_files):
		users_in_range = [uid for uid in userid_list if (uid // increments) == i]

		if not users_in_range:
			continue 

		with open(os.path.join(datapath, 'egodata_' + str(i*increments) + '.tsv'), 'r') as f:
			lines = f.readlines()

		for u in users_in_range:
			u_local = (u % (i*increments) - 1) if i > 0 else u
			l_idx = offset + (u_local)*5
			line = lines[l_idx].strip()
			if line != 'None':
				acts = line.split(' ')
				activity_data[u] = [tuple(map(lambda x: int(x), a.split(':'))) for a in acts]

	return activity_data


if __name__ == '__main__':

	extract_friends()