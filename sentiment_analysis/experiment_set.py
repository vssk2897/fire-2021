from .experiment import CharRNNExperiment
from .extract_features import feature_engg
from .constants import C
import time

class CharRNNExperimentSet :

    def run_experiments(self):
        for language in C.LANGUAGES:
            self._run_experiment_language(language=language)
        return

    
    def _run_experiment_language(self, language) :
        fe = feature_engg(language=language)
        filter_length = [2, 3, 5, 6]
        embedding_size = 256
        lstm_output_size= [512, 256, 128, 64]
        (tdf, vdf) = fe.generate_character_embedding()
        for fil_len in filter_length:
            for lstm_out_size in lstm_output_size:
                experiment = CharRNNExperiment(filter_length=fil_len, embedding_size=embedding_size,\
                    pool_length=fil_len, lstm_output_size=lstm_out_size)
                experiment.run(tdf=tdf, vdf=vdf, language=language)
                time.sleep(100)
    
    def run_sample_experiments(self) :
        filter_len = 6
        embedding_size = 256
        lstm_output_size = 256
        for language in C.LANGUAGES:
            print(language)
            fe = feature_engg(language=language)
            (tdf, vdf) = fe.generate_character_embedding()
            experiment = CharRNNExperiment(filter_length=filter_len, embedding_size=embedding_size,\
                        pool_length=filter_len, lstm_output_size=lstm_output_size)
            experiment.run(tdf=tdf, vdf=vdf, language=language)
