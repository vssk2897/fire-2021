from .experiment import CharRNNExperiment
from .extract_features import feature_engg
from .constants import C
import time
import pandas

class CharRNNExperimentSet :

    def run_experiments(self):
        for language in C.LANGUAGES:
            self._run_experiment_language(language=language)
        return

    
    def _run_experiment_language(self, language) :
        fe = feature_engg(language=language)
        filter_length = [ 5, 6, 7]
        embedding_size = 256
        lstm_output_size= [512, 256, 128]
        (tdf, vdf) = fe.generate_character_embedding()
        del fe
        tdf = pandas.concat([tdf, vdf], ignore_index=True)
        count = 0
        for fil_len in filter_length:
            for lstm_out_size in lstm_output_size:
                experiment = CharRNNExperiment(filter_length=fil_len, embedding_size=embedding_size,\
                    pool_length=fil_len, lstm_output_size=lstm_out_size)
                experiment.run(tdf=tdf, vdf=vdf, language=language)
                print("Sleeping for a minute !!!!")
                time.sleep(60)
    
    def run_v2_experiments(self, path) :
        experiment_list = pandas.read_csv(path).values.tolist()

        for [language, tag] in experiment_list:
            fe = feature_engg(language=language)
            v2_df = fe.get_v2_features()
            del fe
            # tag format [language, self.name + '-v2', self.filter_length, self.embedding_size, self.lstm_output_size, self.pool_length]
            params = tag.split('_')
            filter_length = int(params[2])
            embedding_size = int(params[3])
            lstm_output_size = int(params[4])
            pool_length = int(params[5])
            experiment = CharRNNExperiment(filter_length=filter_length, embedding_size=embedding_size, pool_length=pool_length, lstm_output_size=lstm_output_size)
            experiment.run(tdf = v2_df, vdf=v2_df, language=language, epochs=15, v2=True, batch_size=64)

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
