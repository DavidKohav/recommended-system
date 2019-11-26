from surprise import accuracy
from collections import defaultdict


class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def getTopN(predictions, n=10, minimum_rating=3.0):
        topN = defaultdict(list)

        for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
            if estimated_rating >= minimum_rating:
                topN[int(user_id)].append((int(movie_id), estimated_rating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def hitRate(topNPredicted, left_out_predictions):
        hits = 0
        total = 0

        # For each left-out rating
        for left_out in left_out_predictions:
            user_id = left_out[0]
            left_out_movie_id = left_out[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movie_id, predicted_rating in topNPredicted[int(user_id)]:
                if int(left_out_movie_id) == int(movie_id):
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1

        # Compute overall precision
        return hits / total

    def cumulativeHitRate(topNPredicted, left_out_predictions, rating_cutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for user_id, left_out_movie_id, actual_rating, estimated_rating, _ in left_out_predictions:
            # Only look at ability to recommend things the users actually liked...
            if actual_rating >= rating_cutoff:
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predicted_rating in topNPredicted[int(user_id)]:
                    if int(left_out_movie_id) == movieID:
                        hit = True
                        break
                if hit:
                    hits += 1
                total += 1

        # Compute overall precision
        return hits / total

    def averageReciprocalHitRank(topNPredicted, left_out_predictions):
        summation = 0
        total = 0
        # For each left-out rating
        for user_id, left_out_movie_id, actual_rating, estimated_rating, _ in left_out_predictions:
            # Is it in the predicted top N for this user?
            hit_rank = 0
            rank = 0
            for movie_id, predicted_rating in topNPredicted[int(user_id)]:
                rank = rank + 1
                if int(left_out_movie_id) == movie_id:
                    hit_rank = rank
                    break
            if hit_rank > 0:
                summation += 1.0 / hit_rank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def userCoverage(topNPredicted, num_users, rating_threshold=0):
        hits = 0
        for user_id in topNPredicted.keys():
            hit = False
            for movie_id, predicted_rating in topNPredicted[user_id]:
                if predicted_rating >= rating_threshold:
                    hit = True
                    break
            if hit:
                hits += 1
        print("Done computing user coverage.")
        return hits / num_users

    def novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for user_id in topNPredicted.keys():
            for rating in topNPredicted[user_id]:
                movie_id = rating[0]
                rank = rankings[movie_id]
                total += rank
                n += 1

        print("Done computing novelty.")
        return total / n
