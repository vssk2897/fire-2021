import numpy
import os
from .constants import C
from .models import CharacterRNN
from copy import deepcopy
from keras.preprocessing import sequence
from sklearn.metrics import multilabel_confusion_matrix

class CharRNNExperiment :
    
    def __init__(self, filter_length=3, embedding_size=500, pool_length=3, maxlen=200):
        self.filter_length = filter_length
        self.embedding_size = embedding_size
        self.pool_length = pool_length
        self.maxlen = maxlen
        self.name = 'char-rnn'

    def run(self, tdf, vdf):
        X_train = tdf['char_embedding'].values[:]
        y_train = numpy.array(tdf['char_label'].values[:])
        X_valid = vdf['char_embedding'].values[:]
        y_valid = numpy.array(tdf['char_label'].values[:])
        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        X_valid = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        for language in C.LANGUAGE_TO_ISO_MAPPING.keys():
            tag = '{0}_{1}_{2}_{3}_{4}'.format(language, self.name, self.filter_length, self.embedding_size, self.pool_length)
            RNN = CharacterRNN(language=language, max_features=self.embedding_size, embedding_size=self.embedding_size,\
                filter_length=self.filter_length, nb_filter=self.embedding_size, pool_length=self.filter_length,\
                    lstm_output_size=(self.embedding_size % 128) * 128, epochs=50 )

            print("Starting training of the Char RNN model wit tag {}".format(tag))

            model = RNN.train(deepcopy(X_train), deepcopy(y_train),deepcopy(X_valid),deepcopy(y_valid))
            
            self._save_model(model, tag)

            y_train_pred = model.predict_classes(X_train)
            y_valid_pred = model.predict_classes(X_valid)
            # Schema for an experiment as a row in visualize
            # name, tag, env, accuracy, AUC, multiclass confusion matrix, TPR, other_rate(1-TPR)
            multilabel_confusion_matrix(y_train, y_train_pred)
    
    def _save_model(self, model, tag, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        self._if_experimentdir_exists()
        base_path = os.path.join(Masterdir, Experimentdir, tag)
        if not os.path.isdir(base_path):
            os.mkdir(base_path)
            model_path = os.path.join(base_path, tag+'.bin')
            model.save(model_path)

        return

    def _if_experimentdir_exists(self, Masterdir = os.getcwd(), Experimentdir = C.EXPERIMENT_DIR):
        if os.path.isdir(os.path.join(Masterdir, Experimentdir)) :
            return True
        else:
            os.mkdir(os.path.join(Masterdir, Experimentdir))
            return False



