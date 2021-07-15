import pandas
import numpy
from .constants import C
from .pre_processor import preProcessor
import re
import os
import json
from keras.preprocessing import sequence
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import fasttext
from datasets import load_dataset

class feature_engg:
    
    def __init__(self, language='tamil'):
        self.pre_processor = preProcessor()
        self.language = language
        self.train_df = self.pre_processor.prepare_language_dataset_pandas(language=self.language, env='train')
        self.validation_df = self.pre_processor.prepare_language_dataset_pandas(language=self.language, env='dev')
        self.train_df_character = self.pre_processor.prepare_language_dataset_character(language=self.language, env='train')
        self.validation_df_character = self.pre_processor.prepare_language_dataset_character(language=self.language, env='dev')
        if self._get_old_language_switcher(language=language) is not None:
            self.test_df_character = self.pre_processor.prepare_language_dataset_character_test(self._get_old_language_switcher())
        else:
            self.test_df_character = None

    
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
    
    def generate_character_embedding(self):
        tdf = self.train_df_character.copy(deep=True)
        vdf = self.validation_df_character.copy(deep=True)
        tdf['char_data'] = tdf.apply(lambda x: self._convert_into_character_array(x['tokenized_lines']), axis=1)
        vdf['char_data'] = vdf.apply(lambda x: self._convert_into_character_array(x['tokenized_lines']), axis=1)
        tdf['char_label'] = tdf.apply(lambda x: self._char_level_labels(x['category']), axis=1)
        vdf['char_label'] = vdf.apply(lambda x: self._char_level_labels(x['category']), axis=1)

        model = self._read_char_model('{}_mapping_character_to_number.json'.format(self.language))
        
        if model is not None:
            tdf['char_embedding'] = tdf.apply(lambda x: self._get_character_features(x.char_data, model=model), axis=1)
            vdf['char_embedding'] = vdf.apply(lambda x: self._get_character_features(x.char_data, model=model), axis=1)

        tdf = tdf.drop(columns=['tokenized_lines', 'char_data'])
        vdf = vdf.drop(columns=['tokenized_lines', 'char_data'])
        return (tdf, vdf)

    
    def get_old_test_character_features(self,):
        path = self._get_old_language_switcher()
        if path is None:
            print("For Kannada we do not have test dataset with labels in 2020")
            return None
        test_df = self.pre_processor.prepare_language_dataset_character(path)
        test_df['char_data'] = test_df.apply(lambda x: self._convert_into_character_array(x['tokenized_lines']), axis=1)
        test_df['char_label'] = test_df.apply(lambda x: self._char_level_labels(x['category']), axis=1)
        model = self._read_char_model('{}_mapping_character_to_number.json'.format(self.language))
        if model is not None:
            test_df['char_embedding'] = test_df.apply(lambda x: self._get_character_features(x.char_data, model=model), axis=1)

        test_df = test_df.drop(columns=['tokenized_lines', 'char_data'])
        return test_df

    def get_new_test_character_features(self,):
        path = self._get_new_language_switcher()
        test_df = self.pre_processor.prepare_language_dataset_character(path)
        test_df['char_data'] = test_df.apply(lambda x: self._convert_into_character_array(x['tokenized_lines']), axis=1)
        test_df['char_label'] = test_df.apply(lambda x: self._char_level_labels(x['category']), axis=1)
        model = self._read_char_model('{}_mapping_character_to_number.json'.format(self.language))
        if model is not None:
            test_df['char_embedding'] = test_df.apply(lambda x: self._get_character_features(x.char_data, model=model), axis=1)

        test_df = test_df.drop(columns=['tokenized_lines', 'char_data'])
        return test_df



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
    
    def _convert_into_character_array(self, tokenized_line):
        char_list = []
        for words in tokenized_line:
            for char in words:
                char_list.append(char)
            char_list.append(' ')
        
        return char_list
    
    def _char_level_labels(self, label):
        mapping = C.LANGUAGE_TO_LABEL_MAPPING[self.language]
        return mapping[label]
    
    def _get_character_features(self, char_data, model):
        char_list = list()
        for char in char_data:
            char_list.append([model[char]])
        maxlen=len(model.keys())
        #char_list = sequence.pad_sequences(char_list, maxlen=maxlen)
        return char_list
    
    def _uniq(self, lst):
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item

    def _get_unique(self, l):
        uni = list()
        for row in l:
            uni.extend(list(self._uniq(sorted(row, reverse=True))))
        
        return list(self._uniq(sorted(uni, reverse=True)))


    def train_char_model(self):
        df = None
        for env in ['train', 'dev', 'test']:
            if df is None:
                df = self.pre_processor.prepare_language_dataset_character(language=self.language, env=env)
            else:
                tdf = self.pre_processor.prepare_language_dataset_character(language=self.language, env=env)
                df = pandas.concat([df, tdf], ignore_index=True)
        
        df = df.drop(columns=['category'])
        dataset = load_dataset("offenseval_dravidian", self.language)
        td = dataset['train']['text']
        vd = dataset['validation']['text']
        td = pandas.DataFrame(data=td, columns=['text'])
        vd = pandas.DataFrame(data=vd, columns=['text'])
        offeseval_data = pandas.concat([td, vd], ignore_index=True)
        #offeseval_data = offeseval_data.drop(columns=['idx', 'label'])
        offeseval_data['tokenized_lines'] = offeseval_data.apply(lambda x: self.pre_processor.token( x['text']), axis=1)
        df = pandas.concat([df, offeseval_data], ignore_index=True)
        df['char_data'] = df.apply(lambda x: self._convert_into_character_array(x['tokenized_lines']), axis=1)
        print("Getting unique characters")
        allchars_set = self._get_unique(df.char_data.values)
        for char in allchars_set:
            if char in allchars_set:
                continue
            else:
                allchars_set.add(char)
        

        mapping_char2num = dict()
        mapping_num2char = dict()
        charno = 0
        for char in allchars_set:
            mapping_char2num[char] = charno
            mapping_num2char[charno] = char
            charno += 1

        print('Saving the Character models at {}'.format(os.path.join(os.getcwd(), 'models')))
        f='{}_mapping_character_to_number.json'.format(self.language)
        self._save_char_model(mapping_char2num, filename=f)
        f='{}_mapping_number_to_character.json'.format(self.language)
        self._save_char_model(mapping_num2char, filename=f)
        return (mapping_char2num, mapping_num2char)

    
    def _save_char_model(self, model, masterdir=os.getcwd(), modeldir='models', filename=None) :
        path = os.path.join(masterdir, modeldir, filename)
        with open(path, 'w') as f:
            f.write(json.dumps(model))

    def _read_char_model(self, filename, masterdir=os.getcwd(), modeldir='models') :
        path = os.path.join(masterdir, modeldir, filename)
        with open(path, 'r') as f:
            return json.loads(f.read())
    
    def _get_old_language_switcher(self, language=None):
        switcher = {
			'tamil': os.path.join(os.getcwd() , 'data/2020-dataset/tamil_test_answer.tsv'),
			'kannada': None,
			'malayalam': os.path.join( os.getcwd(), 'data/2020-dataset/malayalam_test_results.tsv')
		}
        if language is not None:
            return switcher.get(language, 'tamil')
        return switcher.get(self.language, 'tamil')

    
    def _get_new_language_switcher(self, language=None):
        switcher = {
            'tamil': os.path.join(os.getcwd(), 'data/2021-dataset/tamil_sentiment_full_test_withoutlabels.tsv'),
            'kannada': os.path.join(os.getcwd(), 'data/2021-dataset/kannada_sentiment_full_test_withoutlabels.tsv'),
            'malayalam': os.path.join(os.getcwd(),'data/2021-dataset/Mal_sentiment_full_test_withoutlabels.tsv' )
        }
        if language is not None:
            return switcher.get(language, 'tamil')
        return switcher.get(self.language, 'tamil')
