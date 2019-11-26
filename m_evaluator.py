from algorithm_manager import AlgorithmManager


class Evaluator:
    algorithms = []

    def __init__(self, movies_content, data):
        movies_content.preparingData(data)
        self.dataset = movies_content

    def addAlgorithm(self, algorithm, name):
        alg = AlgorithmManager(algorithm, name)
        self.algorithms.append(alg)

    def Evaluate(self, do_top_n):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.getName(), "...")
            results[algorithm.getName()] = algorithm.Evaluate(self.dataset, do_top_n)

        # Print results
        print("\n", results)

        if do_top_n:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                    metrics["coverage"], metrics["novelty"]))

        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

    def sampleTopNRecs(self, movie, test_subject=1014, k=10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.getName())

            print("\nBuilding recommendation model...")
            train_set = self.dataset.getFullTrainSet()
            algo.getAlgorithm().fit(train_set)

            print("\nComputing recommendations...")
            test_set = self.dataset.getAntiTestSetForUser(test_subject)

            predictions = algo.getAlgorithm().test(test_set)

            recommendations = []

            print("\nWe recommend For User ", test_subject, " :")
            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                int_movie_id = int(movie_id)
                recommendations.append((int_movie_id, estimated_rating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(movie.getMovieName(ratings[0]), ratings[1])
