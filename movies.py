import os
import csv
import sys
import pandas as pd

from datetime import datetime
from surprise import Dataset
from surprise import Reader
from surprise import KNNBaseline

from collections import defaultdict
from surprise.model_selection import LeaveOneOut
from surprise.model_selection import train_test_split


class MoviesContent:

    movie_id_to_name = {}
    name_to_movie_id = {}
    ratings_path = 'csv_files/ratings.csv'
    movies_path = 'csv_files/movies_metadata.csv'
    
    def __init__(self, preparing, data):
        if preparing:
            self.preparingData(data)

    def loadMovies(self):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        self.movie_id_to_name = {}
        self.name_to_movie_id = {}

        ratings_df = pd.read_csv(self.ratings_path, nrows=100000, low_memory=False)
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        movies_df = pd.read_csv(self.movies_path, low_memory=False)
        movies_df.dropna(subset=['title'], inplace=True)
        movies_df.dropna(subset=['release_date'], inplace=True)

        ## If you want to see better result for user/item base cf please removed the comments.
        #movies_df_unique_ids = movies_df['id'].unique()
        #ratings = ratings_df[ratings_df.movieId.isin(movies_df_unique_ids)]
        #ratings_dataset = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)

        ratings_dataset = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader=reader)

        for index, row in movies_df.iterrows():
            movie_id = int(row['id'])
            movie_name = row['title']

            self.movie_id_to_name[movie_id] = movie_name
            self.name_to_movie_id[movie_name] = movie_id

        return ratings_dataset

    def preparingData(self, data):
        # Build a full training set for evaluating overall properties
        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)

        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test

        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()

        #Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)

        self.rankings = self.getPopularityRanks()

    def getUserRatings(self, user):
        user_ratings = []
        hit_user = False
        with open(self.ratings_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                user_id = int(row[0])
                if user == user_id:
                    movie_id = int(row[1])
                    rating = float(row[2])
                    user_ratings.append((movie_id, rating))
                    hit_user = True
                if hit_user and (user != user_id):
                    break

        return user_ratings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                movie_id = int(row[1])
                ratings[movie_id] += 1
        rank = 1
        for movie_id, rating_count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movie_id] = rank
            rank += 1
        return rankings

    def getMovieName(self, movie_id):
        if movie_id in self.movie_id_to_name:
            return self.movie_id_to_name[movie_id]
        else:
            print(movie_id)
            return ""

    def getMovieID(self, movie_name):

        if movie_name in self.name_to_movie_id:
            return self.name_to_movie_id[movie_name]
        else:
            return 0

    def getGenres(self):
        genres = defaultdict(list)
        genre_ids = {}
        max_genre_id = 0

        movies_df = pd.read_csv(self.movies_path, low_memory=False)
        movies_df.dropna(subset=['title'], inplace=True)
        movies_df.dropna(subset=['release_date'], inplace=True)

        for index, row in movies_df.iterrows():
            movie_id = int(row['id'])
            genre_list = row['genres'].split('|')
            genre_id_list = []
            for genre in genre_list:
                if genre in genre_ids:
                    genre_id = genre_ids[genre]
                else:
                    genre_id = max_genre_id
                    genre_ids[genre] = genre_id
                    max_genre_id += 1
                genre_id_list.append(genre_id)
            genres[movie_id] = genre_id_list
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movie_id, genre_id_list) in genres.items():
            bitfield = [0] * max_genre_id
            for genre_id in genre_id_list:
                bitfield[genre_id] = 1
            genres[movie_id] = bitfield

        return genres

    def getYears(self):
        years = defaultdict(int)
        movies_df = pd.read_csv(self.movies_path, low_memory=False)
        movies_df.dropna(subset=['title'], inplace=True)
        movies_df.dropna(subset=['release_date'], inplace=True)

        for index, row in movies_df.iterrows():
            movie_id = int(row['id'])
            date_object = datetime.strptime(row['release_date'], '%Y-%m-%d').date()
            year = date_object.year
            if year:
                years[movie_id] = int(year)

        return years

    def getFullTrainSet(self):
        return self.fullTrainSet

    def getFullAntiTestSet(self):
        return self.fullAntiTestSet

    def getAntiTestSetForUser(self, test_subject):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(test_subject)
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                         i in trainset.all_items() if
                         i not in user_items]
        return anti_testset

    def getTrainSet(self):
        return self.trainSet

    def getTestSet(self):
        return self.testSet

    def getLOOCVTrainSet(self):
        return self.LOOCVTrain

    def getLOOCVTestSet(self):
        return self.LOOCVTest

    def getLOOCVAntiTestSet(self):
        return self.LOOCVAntiTestSet

    def getPopularityRankings(self):
        return self.rankings
