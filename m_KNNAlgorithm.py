from surprise import AlgoBase
from surprise import PredictionImpossible
from movies import MoviesContent
import numpy as np
import math
import heapq

#Data KNN Algorithm.
class DataKNN(AlgoBase):
    def __init__(self, k=40):
        AlgoBase.__init__(self)
        self.k = k

    # Compute item similarity matrix based on content attributes

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Compute item similarity matrix based on content attributes

        # Load up genre vectors for every movie
        movies = MoviesContent(False, False)
        genres = movies.getGenres()

        print("Computing content-based similarity matrix...")

        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for thisRating in range(self.trainset.n_items):
            if thisRating % 100 == 0:
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating + 1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                if len(genres[thisMovieID]) > 0 and len(genres[otherMovieID]) > 0:
                    genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                    self.similarities[thisRating, otherRating] = genreSimilarity
                    self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
        print("...done.")

        return self

    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]

        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y

        return sumxy / math.sqrt(sumxx * sumyy)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i, rating[0]]
            neighbors.append((genreSimilarity, rating[1]))

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating

        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating

