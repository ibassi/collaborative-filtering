from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import PredictionImpossible

import numpy as np


class MyOwnAlgorithm(AlgoBase):

    def __init__(self, sim_options={}, bsl_options={}):
        AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options)

    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        self.the_mean = np.mean([r for (_, _, r) in
                                 self.trainset.all_ratings()])

        return self

    # def estimate(self, u, i):

    #     return self.the_mean

    # def estimate(self, u, i):
    #
    #     sum_means = self.trainset.global_mean
    #     div = 1
    #
    #     if self.trainset.knows_user(u):
    #         sum_means += np.mean([r for (_, r) in self.trainset.ur[u]])
    #         div += 1
    #     if self.trainset.knows_item(i):
    #         sum_means += np.mean([r for (_, r) in self.trainset.ir[i]])
    #         div += 1
    #
    #     return sum_means / div

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        # Compute similarities between u and v, where v describes all other
        # users that have also rated item i.
        neighbors = [(v, self.sim[u, v]) for (v, r) in self.trainset.ir[i]]
        # Sort these neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        print('The 3 nearest neighbors of user', str(u), 'are:')
        for v, sim_uv in neighbors[:3]:
            print('user {0:} with sim {1:1.2f}'.format(v, sim_uv))

        return sim_uv
        # ... Aaaaand return the baseline estimate anyway ;)

    # def estimate(self, u, i):
	 #    '''Return estimated rating of user u for item i.
	 #       rating_hist is a list of tuples (user, item, rating)'''
    #
	 #    # Retrieve users having rated i
	 #    neighbors = [(sim[u, v], r_vj)
	 #                 for (v, j, r_vj) in rating_hist if (i == j)]
	 #    # Sort them by similarity with u
	 #    neighbors.sort(key=lambda tple: tple[0], reversed=True)
	 #    # Compute weighted average of the k-NN's ratings
	 #    num = sum(sim_uv * r_vi for (sim_uv, r_vi) in neighbors[:k])
	 #    denum = sum(sim_uv for (sim_uv, _) in neighbors[:k])
    #
    # return num / denum

data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

x = cross_validate(algo, data, verbose=True)
