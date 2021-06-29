import pandas
import numpy
from .constants import C
from .pre_processor import preProcessor
import re
import os
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import fasttext

class feature_engg:
    
    def __init__(self, language='tamil'):
        self.pre_processor = preProcessor()
        self.language = language
        self.train_df = self.pre_processor.prepare_language_dataset_pandas(language=language, env='train')
        self.validation_df = self.pre_processor.prepare_language_dataset_pandas(language=language, env='dev')
    
    def generate_word_embedding(self):
        #model = fasttext.load_model("models/fasttext_models/indicnlp.ft.ta.300.bin")
        model = fasttext.load_model(self._get_filename())
        print("Generating word level embedding for {} using fasttext model".format(self.language))
        tdf = self.train_df.copy(deep = True)
        vdf = self.validation_df.copy(deep = True)
        print("Columns {}".format(tdf.columns))
        tdf['transliteration'] = tdf.apply(lambda x: self._transliterate(sentence = x['sentence'], to_scheme= C.LANGUAGES_TO_SANSCRIPT[self.language]), axis=1)                                                                                                                                                                                                                                                                                                                             
        vdf['transliteration'] = vdf.apply(lambda x: self._transliterate(sentence = x['sentence'], to_scheme= C.LANGUAGES_TO_SANSCRIPT[self.language]), axis=1)                                                                                                                                                                                                                                                                                                                             
        tdf['word_vec'] = tdf.apply(lambda x: self._get_word_vector(x['transliteration'], model), axis=1)
        vdf['word_vec'] = vdf.apply(lambda x: self._get_word_vector(x['transliteration'], model), axis=1)
        return (tdf, vdf)
    
    def _get_filename(self):
        key = C.LANGUAGE_TO_ISO_MAPPING[self.language]
        model_base_path = os.path.join(C.MASTER_DIR,C.MODEL_DIR,'fasttext_models')
        path = os.listdir(model_base_path)
        for model in path:
            match = re.match(r'(indicnlp).*({}).*(.bin)'.format(key), model)
            if match:
                #print(match.group())
                return os.path.join(model_base_path, match.group())
        return None
    
    def _transliterate(self, sentence, from_scheme=sanscript.HK, to_scheme=sanscript.TAMIL, pattern =r'^[a-zA-Z]+(\s[a-zA-Z]+)?$') :
        transliteration = list()
        for token in sentence.split() :
            match = re.search(pattern, token)
            if match is not None:
                transliteration.extend([transliterate(match.group(), from_scheme, to_scheme)])
            else:
                transliteration.extend([token])
        return ' '.join(transliteration)

    def _get_word_vector(self, sentence, fasttext_model):
        sentence_vector = list()
        for token in sentence.split():
            sentence_vector.extend([fasttext_model.get_word_vector(token)])
        return numpy.array(sentence_vector)