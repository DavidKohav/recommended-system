from surprise import SVD
from movies import MoviesContent
from m_evaluator import Evaluator
from m_KNNAlgorithm import DataKNN

movies = MoviesContent(False, False)

# Loading movie ratings.
data = movies.loadMovies()

evaluator = Evaluator(movies, data)

# Throw in an SVD recommender
SVDAlgorithm = SVD(random_state=10)
evaluator.addAlgorithm(SVDAlgorithm, "SVD")

data_KNN = DataKNN()
evaluator.addAlgorithm(data_KNN, "dataKNN")

evaluator.Evaluate(True)
evaluator.sampleTopNRecs(movies)



