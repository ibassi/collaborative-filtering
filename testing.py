from surprise import Dataset, evaluate
from surprise import KNNBasic
from collections import defaultdict
import os, io


# from surprise import SVD
# from surprise import Dataset
# from surprise.model_selection import cross_validate


def main():
    data = Dataset.load_builtin("ml-100k")
    trainingSet = data.build_full_trainset()
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }

    knn = KNNBasic(sim_options=sim_options)
    knn.fit(trainingSet)
    testSet = trainingSet.build_anti_testset()
    predictions = knn.test(testSet)

    top3_recommendations = get_top3_recommendations(predictions)
    rid_to_name = read_item_names()
    for uid, user_ratings in top3_recommendations.items():
        print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])


# # Load the movielens-100k dataset (download it if needed),
# data = Dataset.load_builtin('ml-100k')

# # We'll use the famous SVD algorithm.
# algo = SVD()

# # Run 5-fold cross-validation and print results
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

def get_top3_recommendations(predictions, topN=3):
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))

    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recs[uid] = user_ratings[:topN]

    return top_recs


def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and returns a
	mapping to convert raw ids into movie names.
	"""

	file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]

	return rid_to_name


if __name__ == '__main__':
    main()