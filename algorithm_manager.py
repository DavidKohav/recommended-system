from m_recommender import RecommenderMetrics


class AlgorithmManager:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if verbose:
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.getTrainSet())
        predictions = self.algorithm.test(evaluationData.getTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if doTopN:
            # Evaluate top-10 with Leave One Out testing
            if verbose:
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.getLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.getLOOCVTestSet())
            # Build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(evaluationData.getLOOCVAntiTestSet())
            # Compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.getTopN(allPredictions, n)
            if verbose:
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.hitRate(topNPredicted, leftOutPredictions)
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.cumulativeHitRate(topNPredicted, leftOutPredictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.averageReciprocalHitRank(topNPredicted, leftOutPredictions)

            # Evaluate properties of recommendations on full training set
            if verbose:
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.getFullTrainSet())
            all_predictions = self.algorithm.test(evaluationData.getFullAntiTestSet())
            top_n_predicted = RecommenderMetrics.getTopN(all_predictions, n)

            if verbose:
                print("Analyzing coverage and novelty...")

            # Print user coverage with a minimum predicted rating of 3.0:
            metrics["coverage"] = RecommenderMetrics.userCoverage(top_n_predicted,
                                                                  evaluationData.getFullTrainSet().n_users,
                                                                  rating_threshold=3.0)

            # Measure novelty (average popularity rank of recommendations):
            metrics["novelty"] = RecommenderMetrics.novelty(top_n_predicted, evaluationData.getPopularityRankings())

        if verbose:
            print("Analysis complete.")

        return metrics

    def getName(self):
        return self.name

    def getAlgorithm(self):
        return self.algorithm
