from surprise import *
from pprint import pprint
import numpy as np
import pandas as pd

from surprise import SVD, evaluate


def main():
    # Define the format
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    # Load the data from the file using the reader format
    data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)
    data.split(n_folds=6)
    # algo1 = SVD()
    # evaluate(algo1, data, measures=['RMSE', 'MAE'])
    #
    # algo2 = SVDpp()
    # evaluate(algo2, data, measures=['RMSE', 'MAE'])
    #
    # algo3 = KNNBasic()
    # evaluate(algo3, data, measures=['RMSE', 'MAE'])

    # algo4 = NormalPredictor()
    # evaluate(algo4, data, measures=["RMSE", "MAE"])

    # algo4 = KNNWithZScore()
    # evaluate(algo4, data, measures=['RMSE', 'MAE'])

    # algo5 = KNNBaseline()
    # evaluate(algo5, data, measures=['RMSE', 'MAE'])

    algo6 = SlopeOne()
    evaluate(algo6, data, measures=['RMSE', 'MAE'])

    # algo7 = CoClustering()
    # evaluate(algo7, data, measures=['RMSE', 'MAE'])

# ratings = pd.read_csv('ratings_small.csv') # reading data in pandas df

# # to load dataset from pandas df, we need `load_fromm_df` method in surprise lib

# ratings_dict = {'itemID': list(ratings.movieId),
#                 'userID': list(ratings.userId),
#                 'rating': list(ratings.rating)}
# df = pd.DataFrame(ratings_dict)

# # A reader is still needed but only the rating_scale param is required.
# # The Reader class is used to parse a file containing ratings.
# reader = Reader(rating_scale=(0.5, 5.0))

# # The columns must correspond to user id, item id and ratings (in that order).
# data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)


if __name__ == '__main__':
    main()