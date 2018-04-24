from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import PredictionImpossible
import heapq
import numpy as np



class SymmetricAlgo(AlgoBase):
    """This is an abstract class aimed to ease the use of symmetric algorithms.
    A symmetric algorithm is an algorithm that can can be based on users or on
    items indifferently, e.g. all the algorithms in this module.
    When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.
    """

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


    # def __init__(self, bsl_options={}, verbose=True):
    #
    #     AlgoBase.__init__(self, bsl_options=bsl_options)
    #     self.verbose = verbose
    #
    # def fit(self, trainset):
    #
    #     AlgoBase.fit(self, trainset)
    #     self.bu, self.bi = self.compute_baselines()
    #     self.compute_similarities()
    #     # print(self.bu)
    #
    #     return self
    #
    # def estimate(self, u, i):
    #
    #     est = self.trainset.global_mean
    #     if self.trainset.knows_user(u):
    #         est += self.bu[u]
    #     if self.trainset.knows_item(i):
    #         est += self.bi[i]
    #
    #     return est
class KNNBasic(SymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        #for all the  users to have rated the  given item,
        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        neighbors = []
        for (x2, r) in self.yr[y]:
            neighbors.append((self.sim[x, x2],r))

        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


data = Dataset.load_builtin('ml-100k')
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNBasic(sim_options= sim_options)
x = cross_validate(algo, data, verbose=True)

