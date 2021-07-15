from .experiment import CharRNNExperiment
from .extract_features import feature_engg
from .constants import C

class CharRNNExperimentSet :

    def run_experiments(self):
        for language in C.LANGUAGES:
            self.run_experiment_language(language=language)
        return

    
    def run_experiment_language(self, language) :
        fe = feature_engg(language=language)
        filter_length = [2, 3, 5]
        embedding_size = 500
        maxlen = 600
        lstm_output_size= [512, 256, 128]
        (tdf, vdf) = fe.generate_character_embedding()
        for fil_len in filter_length:
            for lstm_out_size in lstm_output_size:
                experiment = CharRNNExperiment(filter_length=fil_len, embedding_size=embedding_size,\
                    pool_length=fil_len, maxlen=maxlen, lstm_output_size=lstm_out_size)
                experiment.run(tdf=tdf, vdf=vdf, language=language)
    
    def run_sample_experiment(self) :
        fe = feature_engg()
        filter_len = 6
        embedding_size = 500
        maxlen = 600
        lstm_output_size = 1024
        for language in C.LANGUAGES:
            (tdf, vdf) = fe.generate_character_embedding(language)
            experiment = CharRNNExperiment(filter_length=filter_len, embedding_size=embedding_size,\
                        pool_length=filter_len, maxlen=maxlen, lstm_output_size=lstm_output_size)
            experiment.run(tdf=tdf, vdf=vdf, language=language)
