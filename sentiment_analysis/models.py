from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.metrics import AUC
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.callbacks import LambdaCallback
import time

from .constants import C
from .extract_features import feature_engg

class CharacterRNN:
    
    def __init__(self, language = 'tamil', max_features=0, embedding_size=128, filter_length= 3, nb_filter=128, pool_length=3, lstm_output_size=128, batch_size = 128, epochs=5, numclasses=C.NUM_OF_CLASSES, maxlen=200) :
        self.language = language
        #LSTM Model Parameters
        # #Embedding
        self.MAX_FEATURES = max_features
        self.embedding_size = embedding_size
        # # Convolution
        self.filter_length = filter_length
        self.nb_filter = nb_filter
        self.pool_length = pool_length
        # LSTM
        self.lstm_output_size = lstm_output_size
        # Training
        self.batch_size = batch_size
        self.number_of_epochs = epochs
        self.numclasses = numclasses
        self.maxlen = maxlen

    def train(self, X_train,y_train,X_valid, y_valid):
        y_train = np_utils.to_categorical(y_train, self.numclasses)
        y_valid = np_utils.to_categorical(y_valid, self.numclasses)
        print('Building  Character level RNN model...')
        model = Sequential()
        model.add(Embedding(self.MAX_FEATURES, self.embedding_size, input_length=self.maxlen))
        model.add(Convolution1D(filters=self.nb_filter,
        					kernel_size=self.filter_length,
							padding='valid',
							activation='relu',
							strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_length))
        model.add(LSTM(units=self.lstm_output_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        model.add(LSTM(units=self.lstm_output_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
        model.add(Dense(self.numclasses))
        model.add(Activation('softmax'))
        # Optimizer is Adamax along with categorical crossentropy loss
        model.compile(loss='categorical_crossentropy',
			  	optimizer='adamax',
			  	metrics=['accuracy', AUC()])
                  
        print('Train ...')
        #Trains model for 50 epochs with shuffling after every epoch for training data and validates on validation data
        model.fit(X_train, y_train, 
			  batch_size=self.batch_size, 
			  shuffle=True, 
			  epochs=self.number_of_epochs,
			  validation_data=(X_valid, y_valid),
              callbacks=[LambdaCallback(on_epoch_begin =  lambda e, l: time.sleep(20) ) ])
        
        return model

    
    def get_old_test_features(self):
        return feature_engg(language=self.language).get_old_test_character_features()
    
    def get_new_test_features(self):
        return 
