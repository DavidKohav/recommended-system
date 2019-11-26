from movies import MoviesContent
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
test_subject = 1014
k = 10

# Load our data set and compute the user similarity matrix
movies = MoviesContent(False, False)
data = movies.loadMovies()

train_set = data.build_full_trainset()
# name: We can use cosine as well.
# user_based we get matrix user to user similarities scores.
sim_options = {'name': 'pearson_baseline',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(train_set)
sims_matrix = model.compute_similarities()

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
test_user_inner_id = train_set.to_inner_uid(test_subject)
similarity_row = sims_matrix[test_user_inner_id]

similar_users = []
for inner_id, score in enumerate(similarity_row):
    if inner_id != test_user_inner_id:
        similar_users.append((inner_id, score))

kNeighbors = heapq.nlargest(k, similar_users, key=lambda t: t[1])

# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similar_user in kNeighbors:
    innerID = similar_user[0]
    user_similarity_score = similar_user[1]
    their_ratings = train_set.ur[innerID]
    for rating in their_ratings:
        candidates[rating[0]] += (rating[1] / 5.0) * user_similarity_score
    
# Build a dictionary of stuff the user has already seen
watched = {}
for item_id, rating in train_set.ur[test_user_inner_id]:
    watched[item_id] = 1
    
# Get top-rated items from similar users:
pos = 0
print("\n")
for item_id, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not item_id in watched:
        movie_id = train_set.to_raw_iid(item_id)
        print(movies.getMovieName(int(movie_id)), rating_sum)
        pos += 1
        if pos > 10:
            break
