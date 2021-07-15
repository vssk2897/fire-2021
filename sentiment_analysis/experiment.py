from builtins import print
import pandas
import numpy
import os
from .constants import C
from .models import CharacterRNN
from copy import deepcopy
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.metrics import multilabel_confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score

class CharRNNExperiment :
    
    def __init__(self, filter_length=3, embedding_size=500, pool_length=3, lstm_output_size=256, maxlen=200):
        self.filter_length = filter_length
        self.embedding_size = embedding_size
        self.pool_length = pool_length
        self.maxlen = maxlen
        self.lstm_output_size= lstm_output_size
        self.name = 'char-rnn'

    def run(self, tdf, vdf, language='tamil'):
        columns = ['name','language' ,'tag', 'env', 'accuracy', 'balanced_accuracy' ,'AUC', 'multiclass_confusion_matrix']
        name = 'char-rnn'
        X_train = tdf['char_embedding'].values[:]
        y_train = numpy.array(tdf['char_label'].values[:])
        X_valid = vdf['char_embedding'].values[:]
        y_valid = numpy.array(tdf['char_label'].values[:])
        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        X_valid = sequence.pad_sequences(X_train, maxlen=self.maxlen)

        tag = '{0}_{1}_{2}_{3}_{4}_{5}'.format(language, self.name, self.filter_length, self.embedding_size, self.lstm_output_size, self.pool_length)
        RNN = CharacterRNN(language=language, max_features=self.embedding_size, embedding_size=self.embedding_size,\
            filter_length=self.filter_length, nb_filter=self.embedding_size, pool_length=self.filter_length,\
                lstm_output_size=self.lstm_output_size,
                epochs=1)

        print("Starting training of the Char RNN model wit tag {}".format(tag))

        model = RNN.train(deepcopy(X_train), deepcopy(y_train),deepcopy(X_valid),deepcopy(y_valid))
        
        self._save_model(model, tag)

        y_train_pred = model.predict_classes(X_train)
        y_valid_pred = model.predict_classes(X_valid)
        
        # Schema for an experiment as a row in visualize
        # name, tag, env, accuracy, balanced_accuracy ,AUC, multiclass confusion matrix, TPR, other_rate(1-TPR)

        evaluation_data = []
        
        evaluation_data.append([name, language ,tag, 'train',
        accuracy_score(y_train, y_train_pred),
        balanced_accuracy_score(y_train, y_train_pred),
        roc_auc_score(y_train, y_train_pred),
        multilabel_confusion_matrix(y_train, y_train_pred)
        ])

        evaluation_data.append([name, language, tag, 'validation',
        accuracy_score(y_valid, y_valid_pred),
        balanced_accuracy_score(y_valid, y_valid_pred),
        roc_auc_score(y_valid, y_valid_pred),
        multilabel_confusion_matrix(y_valid, y_valid_pred)
        ])

        test_df = RNN.get_old_test_features()
        if test_df :
            X_test = sequence.pad_sequences(numpy.array(test_df['char_embedding']), maxlen=self.maxlen)
            y_test = np_utils.to_categorical(numpy.array(test_df['char_label']), self.numclasses)
            y_test_pred = model.predict_classes(X_test)
            evaluation_data.append([name, language,tag, 'test-old',
            accuracy_score(y_test, y_test_pred),
            balanced_accuracy_score(y_test, y_test_pred),
            roc_auc_score(y_test, y_test_pred),
            multilabel_confusion_matrix(y_test, y_test_pred)])
        
        eval_df = pandas.DataFrame(data = evaluation_data)
        self._save_eval_data(eval_df=eval_df, tag=tag)

        test_df = RNN.get_old_test_features()
        if test_df :
            X_test = sequence.pad_sequences(numpy.array(test_df['char_embedding']), maxlen=self.maxlen)
            y_test = np_utils.to_categorical(numpy.array(test_df['char_label']), self.numclasses)
            y_test_pred = model.predict_classes(X_test)
            test_df['sentiments'] = pandas.Series(y_test_pred)
            self._save_final_results(test_df=test_df)

            
    def _save_final_results(self, test_df, tag, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        base_path = os.path.join(Masterdir, Experimentdir, tag)
        data_path = os.path.join(base_path, tag + '_results'+'.tsv')
        print('Saving results for the language for tag {}'.format(tag, data_path)) 
        test_df.to_csv(data_path, index=False, sep=C.SEPERATOR)

    def _save_eval_data(self, eval_df, tag, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        base_path = os.path.join(Masterdir, Experimentdir, tag)
        data_path = os.path.join(base_path, tag +'_eval' +'.tsv')
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

    def _if_experimentdir_exists(self, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        if os.path.isdir(os.path.join(Masterdir, Experimentdir)) :
            return True
        else:
            os.mkdir(os.path.join(Masterdir, Experimentdir))
            return False



