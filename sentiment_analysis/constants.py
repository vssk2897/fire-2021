class C:
    MASTER_DIR = '/home/vsskarthik/research-project/fire-2021'
    DATA_DIR = 'data/'
    MODEL_DIR = 'models/'
    FEATURE_DIR = 'features/'
    ENVIRONMENTS = ['train', 'dev']
    #Data I/O formatting
    SEPERATOR = '\t'
    DATA_COLUMN = 0
    LABEL_COLUMN = 1
    LABELS_TAMIL =  {'positive ': 4, 'negative ': 3, 'mixed_feelings ': 2, 'unknown_state ': 1,'not-tamil ': 0}# 0 -> Negative, 1-> Neutral, 2-> Positive
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

