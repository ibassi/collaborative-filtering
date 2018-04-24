import numpy as np
import pandas as pd

from sklearn import cross_validation as cv
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import euclidean_distances as ed

from sklearn.metrics import pairwise_distances

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from math import sqrt


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test


# def train_test_split(*arrays, **options):
#     """Split arrays or matrices into random train and test subsets
#
#     Quick utility that wraps input validation and
#     ``next(ShuffleSplit().split(X, y))`` and application to input data
#     into a single call for splitting (and optionally subsampling) data in a
#     oneliner.
#
#     Read more in the :ref:`User Guide <cross_validation>`.
#
#     Parameters
#     ----------
#     *arrays : sequence of indexables with same length / shape[0]
#         Allowed inputs are lists, numpy arrays, scipy-sparse
#         matrices or pandas dataframes.
#
#     test_size : float, int, None, optional
#         If float, should be between 0.0 and 1.0 and represent the proportion
#         of the dataset to include in the test split. If int, represents the
#         absolute number of test samples. If None, the value is set to the
#         complement of the train size. By default, the value is set to 0.25.
#         The default will change in version 0.21. It will remain 0.25 only
#         if ``train_size`` is unspecified, otherwise it will complement
#         the specified ``train_size``.
#
#     train_size : float, int, or None, default None
#         If float, should be between 0.0 and 1.0 and represent the
#         proportion of the dataset to include in the train split. If
#         int, represents the absolute number of train samples. If None,
#         the value is automatically set to the complement of the test size.
#
#     random_state : int, RandomState instance or None, optional (default=None)
#         If int, random_state is the seed used by the random number generator;
#         If RandomState instance, random_state is the random number generator;
#         If None, the random number generator is the RandomState instance used
#         by `np.random`.
#
#     shuffle : boolean, optional (default=True)
#         Whether or not to shuffle the data before splitting. If shuffle=False
#         then stratify must be None.
#
#     stratify : array-like or None (default is None)
#         If not None, data is split in a stratified fashion, using this as
#         the class labels.
#
#     Returns
#     -------
#     splitting : list, length=2 * len(arrays)
#         List containing train-test split of inputs.
#
#         .. versionadded:: 0.16
#             If the input is sparse, the output will be a
#             ``scipy.sparse.csr_matrix``. Else, output type is the same as the
#             input type.
#
#     Examples
#     --------
#     [[0, 1, 2], [3, 4]]
#
#     """
#     n_arrays = len(arrays)
#     if n_arrays == 0:
#         raise ValueError("At least one array required as input")
#     test_size = options.pop('test_size', 'default')
#     train_size = options.pop('train_size', None)
#     random_state = options.pop('random_state', None)
#     stratify = options.pop('stratify', None)
#     shuffle = options.pop('shuffle', True)
#
#     if options:
#         raise TypeError("Invalid parameters passed: %s" % str(options))
#
#     if test_size == 'default':
#         test_size = None
#         if train_size is not None:
#             warnings.warn("From version 0.21, test_size will always "
#                           "complement train_size unless both "
#                           "are specified.",
#                           FutureWarning)
#
#     if test_size is None and train_size is None:
#         test_size = 0.25
#
#     arrays = indexable(*arrays)
#
#     if shuffle is False:
#         if stratify is not None:
#             raise ValueError(
#                 "Stratified train/test split is not implemented for "
#                 "shuffle=False")
#
#         n_samples = _num_samples(arrays[0])
#         n_train, n_test = _validate_shuffle_split(n_samples, test_size,
#                                                   train_size)
#
#         train = np.arange(n_train)
#         test = np.arange(n_train, n_train + n_test)
#
#     else:
#         if stratify is not None:
#             CVClass = StratifiedShuffleSplit
#         else:
#             CVClass = ShuffleSplit
#
#         cv = CVClass(test_size=test_size,
#                      train_size=train_size,
#                      random_state=random_state)
#
#         train, test = next(cv.split(X=arrays[0], y=stratify))
#
#     return list(chain.from_iterable((safe_indexing(a, train),
#                                      safe_indexing(a, test)) for a in arrays))


def user_similarity_matrix(ratings):
    # epsilon -> small number for handling dived-by-zero errors
    epsilon = 1e-9
    sim = ratings.dot(ratings.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def item_similarity_matrix(ratings):
    epsilon = 1e-9
    sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def cosine_similarity(ratings):
    # return np.dot(ratings, ratings.T) / (np.sqrt(np.dot(ratings, ratings)) * np.sqrt(np.dot(ratings.T, ratings.T)))

    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    epsilon = 1e-9
    similarity = ratings.dot(ratings.T) + epsilon

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

def euclidean_similarity(ratings):
    return ed(ratings)


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def predict_nobias(ratings, similarity, kind='user'):
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        pred = (similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T)
        pred += user_bias[:, np.newaxis]
    elif kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]

    return pred


# def predict_nobias2(ratings, similarity, kind='user'):
#     if kind == 'user':
#         user_bias = ratings.mean(axis=1)
#         user_std = ratings.std(axis=1)
#
#         ratings = (ratings - user_bias[:, np.newaxis]).copy()
#         pred = user_std[:, np.newaxis]((similarity.dot(ratings) /user_std[:, np.newaxis]) / np.array([np.abs(similarity).sum(axis=1)]).T)
#         pred += user_bias[:, np.newaxis]
#     elif kind == 'item':
#         item_bias = ratings.mean(axis=0)
#         ratings = (ratings - item_bias[np.newaxis, :]).copy()
#         pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
#         pred += item_bias[np.newaxis, :]
#
#     return pred


def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':

        print("similarity.dot(ratings)")
        print(similarity.dot(ratings))
        print("np.array([np.abs(similarity).sum(axis=1)]).T")
        print(np.array([np.abs(similarity).sum(axis=1)]).T.shape)
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        print(pred.shape)
        print(pred)
        return pred
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        k = 15
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        k = 40
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

    return pred



# def predict_topk_nobias(ratings, similarity, kind='user', k=40):
#     pred = np.zeros(ratings.shape)
#     if kind == 'user':
#         user_bias = ratings.mean(axis=1)
#         ratings = (ratings - user_bias[:, np.newaxis]).copy()
#         for i in range(ratings.shape[0]):
#             top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
#             for j in range(ratings.shape[1]):
#                 pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
#                 pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
#         pred += user_bias[:, np.newaxis]
#     if kind == 'item':
#         item_bias = ratings.mean(axis=0)
#         ratings = (ratings - item_bias[np.newaxis, :]).copy()
#         for j in range(ratings.shape[1]):
#             top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
#             for i in range(ratings.shape[0]):
#                 pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
#                 pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
#         pred += item_bias[np.newaxis, :]
#
#     return pred


def get_rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def get_mae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction, ground_truth)


def main():
    setup = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=setup)

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    # train, test = cv.train_test_split(df, test_size=0.25)
    train, test = ms.train_test_split(df, train_size=0.90)

    train_matrix = np.zeros((n_users, n_items))
    for line in train.itertuples():
        train_matrix[line[1]-1, line[2]-1] = line[3]

    # print(train_matrix.shape[1])

    #fill in matrix with test data
    test_matrix = np.zeros((n_users, n_items))
    for line in test.itertuples():
        test_matrix[line[1]-1, line[2]-1] = line[3]

    # train, test = train_test_split(ratings)

    # user_similarity = user_similarity_matrix(train_matrix)
    # item_similarity = item_similarity_matrix(train_matrix)

    user_similarity = cosine_similarity(train_matrix)
    item_similarity = cosine_similarity(train_matrix.T)

    print(user_similarity.shape)
    print(item_similarity.shape)

    # user_similarity = pairwise_distances(train_matrix, metric="cosine")
    # item_similarity = pairwise_distances(train_matrix.T, metric="cosine")
    print("user_similarity")
    print(user_similarity)
    # print("item_similarity")
    # print(item_similarity)

    item_prediction = predict_topk(train_matrix, item_similarity, kind='item')
    user_prediction = predict_topk(train_matrix, user_similarity, kind='user')

    print('User-based CF RMSE: ' + str(get_rmse(user_prediction, test_matrix)))
    print('Item-based CF RMSE: ' + str(get_rmse(item_prediction, test_matrix)))

    print('User-based CF MAE: ' + str(get_mae(user_prediction, test_matrix)))
    print('Item-based CF MAE: ' + str(get_mae(item_prediction, test_matrix)))

    print('User-based CF MSE: ' + str(get_mse(user_prediction, test_matrix)))
    print('Item-based CF MSE: ' + str(get_mse(item_prediction, test_matrix)))


if __name__ == '__main__':
    main()