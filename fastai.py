

# http://surprise.readthedocs.io/en/stable/getting_started.html
# I believe in loading all the datasets from pandas df 
# you can also load dataset from csv and whatever suits
# import sys
# sys.path.append("/Users/ibassi/fastai") 
# from surprise import Reader, Dataset
import pandas as pd
import numpy as np
from fastai.learner import *
from fastai.column_data import *


def main():
	ratings = pd.read_csv('./ml-latest-small/ratings.csv') # loading data from csv
	"""
	ratings_small.csv has 4 columns - userId, movieId, ratings, and timestammp
	it is most generic data format for CF related data
	"""

	val_indx = get_cv_idxs(len(ratings))  # index for validation set
	wd = 2e-4 # weight decay
	n_factors = 50 # n_factors - dimension of embedding matrix (D)

	# data loader
	cf = CollabFilterDataset.from_csv(path, 'ratings_small.csv', 'userId', 'movieId', 'rating')

	# learner initializes model object
	learn = cf.get_learner(n_factors, val_indx, bs=64, opt_fn=optim.Adam)

	# fitting model with 1e-2 learning rate, 2 epochs, 
	# (1 cycle length and 2 cycle multiple for learning rate scheduling)
	learn.fit(1e-2,2, wds = wd, cycle_len=1, cycle_mult=2)


if 	__name__ == '__main__':
	main()