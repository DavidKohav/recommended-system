from movies import MoviesContent
from surprise import KNNBasic
from collections import defaultdict
from operator import itemgetter
        
test_subject = 1014
k = 10

movies = MoviesContent(False, False)
data = movies.loadMovies()

train_set = data.build_full_trainset()

sim_options = {'name': 'pearson_baseline',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(train_set)
sims_matrix = model.compute_similarities()

test_user_inner_id = train_set.to_inner_uid(test_subject)

# Get the top K items we rated.
test_user_ratings = train_set.ur[test_user_inner_id]

print("\n")
kNeighbors = []
for rating in test_user_ratings:
    if rating[1] > 4.0:
        kNeighbors.append(rating)

# Get similar items to stuff we liked (weighted by rating).
candidates = defaultdict(float)
for item_id, rating in kNeighbors:
    similarity_row = sims_matrix[item_id]
    for innerID, score in enumerate(similarity_row):
        candidates[innerID] += score * (rating / 5.0)
    
# Build a dictionary of stuff the user has already seen.
watched = {}
for item_id, rating in train_set.ur[test_user_inner_id]:
    watched[item_id] = 1
    
# Get top-rated items from similar users:
pos = 0
for item_id, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not item_id in watched:
        movie_id = train_set.to_raw_iid(item_id)
        print(movies.getMovieName(int(movie_id)), rating_sum)
        pos += 1
        if pos > 10:
            break
