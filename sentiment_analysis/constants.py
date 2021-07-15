from indic_transliteration import sanscript
import os
class C:
    
    #MASTER_DIR = '/home/vsskarthik/research-project/fire-2021'
    MASTER_DIR = os.getcwd()
    DATA_DIR = 'data/'
    MODEL_DIR = 'models/'
    FEATURE_DIR = 'features/'
    EXPERIMENT_DIR = 'experiments/'
    ENVIRONMENTS = ['train', 'dev']
    
    #Data I/O formatting
    SEPERATOR = '\t'
    DATA_COLUMN = 0
    LABEL_COLUMN = 1
    LANGUAGES = [ 'tamil','kannada','malayalam' ]
    _2021_dataset = '2021-dataset'
    _2020_dataset = '2020-dataset'

    NUM_OF_CLASSES = 5
    
    # TODO: Will be converting all the column names in the respective laguages to the lower case and remove the trailing spaces
    LABELS_TAMIL =  {'Positive': 4, 'Negative': 3, 'Mixed_feelings': 2, 'unknown_state': 1,'not-Tamil': 0}
    LABELS_MALAYALAM = {'Positive': 4, 'Negative': 3, 'Mixed_feelings': 2, 'unknown_state': 1,'not-malayalam': 0, 'Mixed feelings': 2, 'unknown state': 1 }
    LABELS_KANNADA = {'Positive': 4, 'Negative': 3, 'Mixed_feelings': 2, 'unknown_state': 1,'not-Kannada': 0, 'Mixed feelings': 2, 'unknown state': 1 }
    
    # Labels Inverse mapping
    LABELS_TAMIL_INVERSE =  {4 : 'Positive',  3 : 'Negative', 2: 'Mixed_feelings', 1: 'unknown_state' , 0 : 'not-Tamil' }
    LABELS_MALAYALAM_INVERSE = {4 : 'Positive',  3 : 'Negative', 2: 'Mixed_feelings', 1: 'unknown_state' , 0 :'not-malayalam' }
    LABELS_KANNADA_INVERSE = {4 : 'Positive',  3 : 'Negative',  2 : 'Mixed feelings',  1 : 'unknown state',  0 :'not-Kannada' }
    LABELS_INVERSE = { 'tamil': LABELS_TAMIL_INVERSE, 'malayalam': LABELS_MALAYALAM_INVERSE, 'kannada': LABELS_KANNADA_INVERSE}
    

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

