from sentiment_analysis.extract_features import feature_engg
from builtins import print
import pandas
import numpy
import os
from .constants import C
from .models import CharacterRNN
from copy import deepcopy
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score

class CharRNNExperiment :
    
    def __init__(self, filter_length=3, embedding_size=256, pool_length=3, lstm_output_size=256):
        self.filter_length = filter_length
        self.embedding_size = embedding_size
        self.pool_length = pool_length
        self.lstm_output_size= lstm_output_size
        self.name = 'char-rnn'

    def run(self, tdf, vdf, language='tamil', v2=False, epochs=40, batch_size=256):
        
        columns = ['name','language' ,'tag', 'env', 'accuracy', 'balanced_accuracy' , 'multiclass_confusion_matrix']
        name = 'char-rnn'
        fe = feature_engg(language=language)
        char_model = fe._read_char_model(filename='{}_mapping_character_to_number.json'.format(language))
        max_features = len(char_model.keys())
        maxlen = max_features
        print(maxlen)
        del fe
        
        X_train = sequence.pad_sequences(tdf['char_embedding'].values[:], maxlen=self.embedding_size)
        y_train = tdf['char_label'].values[:]
        X_valid = sequence.pad_sequences(vdf['char_embedding'].values[:], maxlen=self.embedding_size)
        y_valid = vdf['char_label'].values[:]
        if v2:
            tag = '{0}_{1}_{2}_{3}_{4}_{5}'.format(language, self.name + '-v2', self.filter_length, self.embedding_size, self.lstm_output_size, self.pool_length)
        else:
            tag = '{0}_{1}_{2}_{3}_{4}_{5}'.format(language, self.name, self.filter_length, self.embedding_size, self.lstm_output_size, self.pool_length)
        RNN = CharacterRNN(language=language, max_features=max_features, embedding_size=self.embedding_size,\
            filter_length=self.filter_length, nb_filter=self.embedding_size, pool_length=self.filter_length,\
                lstm_output_size=self.lstm_output_size, epochs=epochs, maxlen=maxlen, batch_size=batch_size)

        model = None
        

        if self._is_model(tag) and not v2:
            base_path = os.path.join(C.MASTER_DIR, C.EXPERIMENT_DIR, tag)
            path = os.path.join(base_path, tag+'.h5')
            model = load_model(path)

        elif v2:
            base_path = os.path.join(C.MASTER_DIR, C.EXPERIMENT_DIR, tag)
            
            path = os.path.join(base_path, tag+'.h5')
            model = load_model(path)
            


        else :
            print("Starting training of the Char RNN model with tag {}".format(tag))
            model = RNN.train(deepcopy(X_train), deepcopy(y_train),deepcopy(X_valid),deepcopy(y_valid))
            self._save_model(model, tag)

        #     y_train_pred = model.predict_classes(X_train)
        y_valid_pred = model.predict_classes(X_valid)
        
        # Schema for an experiment as a row in visualize
        # name, tag, env, accuracy, balanced_accuracy ,AUC, multiclass confusion matrix, TPR, other_rate(1-TPR)

        evaluation_data = []
        """
        evaluation_data.append([name, language ,tag, 'train',
        accuracy_score(y_train, y_train_pred),
        balanced_accuracy_score(y_train, y_train_pred),
        roc_auc_score(y_train, y_train_pred, multi_class='ovo'),
        multilabel_confusion_matrix(y_train, y_train_pred)
        ])
        """

        evaluation_data.append([name, language, tag, 'validation',
        accuracy_score(y_valid, y_valid_pred),
        balanced_accuracy_score(y_valid, y_valid_pred),
        multilabel_confusion_matrix(y_valid, y_valid_pred)
        ])

        test_df = RNN.get_old_test_features()
        if test_df is not None :
            X_test = sequence.pad_sequences(numpy.array(test_df['char_embedding']), maxlen=maxlen)
            y_test = numpy.array(test_df['char_label'])
            y_test_pred = model.predict_classes(X_test)
            evaluation_data.append([name, language,tag, 'test-old',
            accuracy_score(y_test, y_test_pred),
            balanced_accuracy_score(y_test, y_test_pred),
            multilabel_confusion_matrix(y_test, y_test_pred)])
        
        eval_df = pandas.DataFrame(data = evaluation_data, columns=columns)
        self._save_eval_data(eval_df=eval_df, tag=tag)

        test_df = RNN.get_new_test_features()
        if test_df is not None :
            X_test = sequence.pad_sequences(numpy.array(test_df['char_embedding']), maxlen=maxlen)
            y_test_pred = model.predict_classes(X_test)
            test_df['sentiments'] = pandas.Series(y_test_pred)
            test_df['category'] = test_df.apply(lambda x : self._inverse_labels(x, language), axis = 1)
            test_df = test_df.drop(columns=['char_embedding', 'sentiments'])
            self._save_final_results(test_df=test_df, tag=tag)

    
    def _get_language_labels(self, language):
        return C.LABELS_INVERSE[language]

    
    def _inverse_labels(self, row,language):
        return self._get_language_labels(language)[row['sentiments']]
            

            
    def _save_final_results(self, test_df, tag, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        base_path = os.path.join(Masterdir, Experimentdir, tag)
        data_path = os.path.join(base_path, tag + '_results'+'.tsv')
        if os.path.isfile(data_path):
            data_path = os.path.join(base_path, tag + '_v2'+ '_results'+'.tsv')
        print('Saving results for the language for tag {}'.format(tag, data_path)) 
        test_df.to_csv(data_path, index=False, sep=C.SEPERATOR)

    def _save_eval_data(self, eval_df, tag, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        base_path = os.path.join(Masterdir, Experimentdir, tag)
        data_path = os.path.join(base_path, tag +'_eval' +'.tsv')
        if os.path.isfile(data_path):
            data_path = os.path.join(base_path, tag + '_v2'+ '_eval' +'.tsv')
        print('Saving evaluation data for tag {}  in the data path {}'.format(tag, data_path))
        eval_df.to_csv(data_path, index=False, sep=C.SEPERATOR)


    def _save_model(self, model, tag, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        self._if_experimentdir_exists()
        base_path = os.path.join(Masterdir, Experimentdir, tag)
        if not os.path.isdir(base_path):
            os.mkdir(base_path)
            model_path = os.path.join(base_path, tag+'.h5')
            print('Saving the model with tag {} at path {}'.format(tag, model_path))
            model.save(model_path)

        else :
            model_path = os.path.join(base_path, tag+'.h5')
            print('Saving the model with tag {} at path {}'.format(tag, model_path))
            model.save(model_path)
    
    def _is_model(self, tag, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        base_path = os.path.join(Masterdir, Experimentdir, tag)
        if os.path.isfile(os.path.join(base_path, tag+'.h5')):
            return True
        else:
            return False



    def _if_experimentdir_exists(self, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        if os.path.isdir(os.path.join(Masterdir, Experimentdir)) :
            return True
        else:
            os.mkdir(os.path.join(Masterdir, Experimentdir))
            return False



