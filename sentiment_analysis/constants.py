from indic_transliteration import sanscript
import os
class C:
    #MASTER_DIR = '/home/vsskarthik/research-project/fire-2021'
    MASTER_DIR = os.getcwd()
    DATA_DIR = 'data/'
    MODEL_DIR = 'models/'
    FEATURE_DIR = 'features/'
    ENVIRONMENTS = ['train', 'dev']
    #Data I/O formatting
    SEPERATOR = '\t'
    DATA_COLUMN = 0
    LABEL_COLUMN = 1
    LANGUAGES = [ 'tamil','kannada','malayalam' ]
    # TODO: Will be converting all the column names in the respective laguages to the lower case and remove the trailing spaces
    LABELS_TAMIL =  {'Positive': 4, 'Negative': 3, 'Mixed_feelings': 2, 'unknown_state': 1,'not-Tamil': 0}# 0 -> Negative, 1-> Neutral, 2-> Positive
    LABELS_MALAYALAM = {'Positive': 4, 'Negative': 3, 'Mixed_feelings': 2, 'unknown_state': 1,'not-malayalam': 0}
    LABELS_KANNADA = {'Positive': 4, 'Negative': 3, 'Mixed_feelings': 2, 'unknown_state': 1,'not-Kannada': 0}
    LANGUAGE_TO_ISO_MAPPING = {
        'tamil': 'ta',
        'kannada': 'kn',
        'malayalam': 'ml'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    }
    LANGUAGES_TO_SANSCRIPT = {
        'tamil': sanscript.TAMIL,
        'kannada': sanscript.KANNADA,
        'malayalam': sanscript.MALAYALAM
    }
    LANGUAGE_TO_LABEL_MAPPING = {
        'tamil': LABELS_TAMIL,
        'malayalam': LABELS_MALAYALAM,
        'kannada': LABELS_KANNADA
    }
    EXPERIMENT_DIR = 'experiments/'
    MAXLEN = 200
    _2021_dataset = '2021-dataset'
    _2020_dataset = '2020-dataset'
    #LSTM Model Parameters
    #Embedding
    MAX_FEATURES = 0
    EMBEDDING_SIZE = 128
    # Convolution
    FILTER_LENGTH = 3
    NB_FILTER = 128
    POOL_LENGTH = 3
    # LSTM
    LSTM_OUTPUT_SIZE = 128
    # Training
    BATCH_SIZE = 128
    NUMBER_OF_EPOCHS = 1
    NUM_OF_CLASSES = 5

