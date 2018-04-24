from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import PredictionImpossible
from surprise.model_selection.split import KFold

import numpy as np


class MyOwnAlgorithm(AlgoBase):

    def __init__(self, sim_options={}, bsl_options={}):
        AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options)
        self.trainset.rating_scale = (1,5)

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        self.similarity_matrix = self.compute_similarities()

        return self

    def estimate(self, u, i):
        predicted = current = max = 0


        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        #for all the users to have rated the given item,
        neighbors = [(self.similarity_matrix[x, x2], r) for (x2, r) in self.yr[y]]

        for i in range(self.trainset.rating_scale[0], self.trainset.rating_scale[1]):
            print(i)

        return 3




data = Dataset.load_builtin('ml-100k')
kf = KFold(n_splits=3)

algo = MyOwnAlgorithm(kf)

x = cross_validate(algo, data, verbose=True)
