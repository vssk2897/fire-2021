from sentiment_analysis import CharRNNExperimentSet

exp_set = CharRNNExperimentSet()
#exp_set.run_experiments()
exp_set.run_v2_experiments(path = 'v2_experiments.csv')
