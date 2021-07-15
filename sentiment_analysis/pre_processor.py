import re
from sys import path
import numpy
import os
import pandas
from .constants import C

class preProcessor:
	"""This class is resposible for getting raw data from the data store
	"""

	def token(self, sentence, remove_repeat=False, minchars=2):
		""" Tokenizes the sentence passed onto the function
			Idea is to remove the repeat characters from the sentence like
			'superrrrrr   suriyaaaaa  luv u   all  d best' and transform it into 



		Args:
			sentence (string, mandatory): [description]
			remove_repeat (bool, optional): Remove repeat characters. Defaults to False.
			minchars (int, optional): minimum number of characters for a token. Defaults to 2.

		Returns:
			[list of string]: all the tokens in the list form
		"""
		tokens = []
		if remove_repeat:
			sentence = self._removeRepeat(sentence)

		for t in sentence.lower().rstrip().split():
			if len(t)>=minchars:
				tokens.append(t)
		return tokens

	def _removeRepeat(self, string):
		"""Removes the repeated characters from the given sentence with the help of regular expression

		Args:
			string (string): sentence from the dataset

		Returns:
			string: sentence with repeated characters removed
		"""
		return re.sub(r'(.)\1+', r'\1\1', string)
	
	def parse(self, path=None, Masterdir=None, Datadir=None, filename=None, seperator='\t', datacol=C.DATA_COLUMN, labelcol=C.LABEL_COLUMN, labels= C.LABELS_TAMIL):
		"""Parses the dataset file returns the raw data along with labels

		Args:
			Masterdir ([type]): MASTER DIRECTORY
			Datadir ([type]): Data directory
			filename ([type]): data file name
			seperator ([type]): tsv so \t
			datacol ([type]): Column  name
			labelcol ([type]): Column name
			labels ([type]): Dictionary of labels

		Returns:
			[list]: Return list of size two where second variable is numpy array
		"""
    	#Reads the files and splits data into individual lines
		print(path)
		if path is not None and os.path.isfile(path):
			f = open(path, 'r')
		else: 

			f = open(Masterdir + Datadir + filename, 'r')
		orig_lines = f.read()
		orig_lines = orig_lines.split('\n')[1 : -1]
		lines = f.read().lower()
		lines = lines.lower().split('\n')[1 : -1]
		print('Number of rows {}'.format(len(lines)))

		X_train = []
		Y_train = []
		orig_text = []
		for line in orig_lines:
			line = line.split(seperator)
			orig_text.extend([line[0]])
		#Processes individual lines
		for line in lines:
			# Seperator for the current dataset. Currently '\t'. 
			line = line.split(seperator)
			#Token is the function which implements basic preprocessing as mentioned in our paper
			tokenized_lines = self.token(line[datacol], remove_repeat=True)
			
			#Creates character lists
			char_list = []
			for words in tokenized_lines:
				for char in words:
					char_list.append(char)
				char_list.append(' ')
			#print(char_list) - Debugs the character list created
			X_train.append(char_list)
			
			#Appends labels
			if line[labelcol] in labels.keys():
				Y_train.append(labels[line[labelcol]])
			else:
				print("Error occurred while parsing the dataset for the line " + str(line))
		#Convert X-train & Y_train to a numpy array	
		Y_train = numpy.asarray(Y_train)
		assert(len(X_train) == Y_train.shape[0])
		#X_train = numpy.asarray(X_train)
		print("Length of data {}".format(len(X_train)) )
		return (orig_text ,X_train,Y_train)

	def _get_language_dataset_filenames(self, masterdir=os.getcwd(), datadir=C.DATA_DIR, language='tamil', data_env='train'):
		"""Returns the file names for the  corresponding languagedataset 

		Args:
			masterdir ([string], optional): [description]. Defaults to os.getcwd().
			datadir ([string], optional): [description]. Defaults to C.DATA_DIR.

		Returns:
			[list]: list of language dataset filenames
		"""
		base_data_path = os.path.join(masterdir, datadir)
		fire_datadirs = os.listdir(base_data_path)
		only_files = list()
		for datadir in fire_datadirs:
			mypath = os.path.join(base_data_path, datadir)
			for name in os.listdir(mypath):
				match = re.match(self._get_language_switcher(language, data_env), name)
				if match:
					if len(match.groups()) >= 3:
						only_files.extend([os.path.join(mypath, name)])
			#only_files.extend([f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))])
		return only_files

	def _get_language_switcher(self, language, data_env='train'):
		switcher = {
			'tamil': r'(tamil).*({0}).*(.tsv)'.format(data_env),
			'kannada': r'(kannada).*({0}).*(.tsv)'.format(data_env),
			'malayalam': r"(Mal).*({0}).*(.tsv)|(malayalam).*({1}).*(.tsv)".format(data_env, data_env)
		}
		return switcher.get(language, 'tamil')
	
	def prepare_language_dataset_pandas(self, language='tamil', env='train'):
		df = None
		for f in self._get_language_dataset_filenames(masterdir=C.MASTER_DIR, datadir=C.DATA_DIR, language=language, data_env=env):
			if df is None:
				tdf = pandas.read_csv(f, sep=C.SEPERATOR)
				tdf['category'] =  tdf['category'].str.rstrip()
				#print(f)
				#print(tdf.columns)
				tdf['sentence'] = tdf.apply(lambda x: self._removeRepeat(x['text']), axis=1)
				#print(tdf['category'].value_counts())
				df = pandas.concat([df, tdf], ignore_index=True)
			else:
				df = pandas.read_csv(f, sep=C.SEPERATOR)
				#print(df.columns)
				df['sentence'] = df.apply(lambda x: self._removeRepeat(x['text']), axis=1)
				df['category'] =  df['category'].str.rstrip()
			
		#print(df.columns)
		return df

	def prepare_language_dataset_character(self, language='tamil', env='train') :
		df = None
		for f in self._get_language_dataset_filenames(masterdir=C.MASTER_DIR, datadir=C.DATA_DIR, language=language, data_env=env):
			if df is not None:
				tdf = pandas.read_csv(f, sep=C.SEPERATOR)
				col = tdf.columns.difference(['text'])
				tdf['category'] =  tdf[col[0]].str.rstrip()
				#tdf['category'] =  tdf['category'].str.rstrip()
				tdf['tokenized_lines'] = tdf.apply(lambda x: self.token(x['text'], remove_repeat=True), axis=1)
				#print(tdf['category'].value_counts())
				df = pandas.concat([df, tdf], ignore_index=True)
			else:
				df = pandas.read_csv(f, sep=C.SEPERATOR)
				#print(df.columns)
				col = df.columns.difference(['text'])
				df['category'] =  df[col[0]].str.rstrip()
				df['tokenized_lines'] = df.apply(lambda x: self.token(x['text'], remove_repeat=True), axis=1)
				

		return df

	def prepare_language_dataset_character_test(self, path):
		df = None
		df = pandas.read_csv(path, sep=C.SEPERATOR)
		col = df.columns.difference(['text'])
		df['category'] =  df[col[0]].str.rstrip()
		df['tokenized_lines'] = df.apply(lambda x: self.token(x['text'], remove_repeat=True), axis=1)
		return df

	def prepare_language_dataset_character_test_new(self, path):
		df = None
		df = pandas.read_csv(path, sep=C.SEPERATOR)
		df['tokenized_lines'] = df.apply(lambda x: self.token(x['text'], remove_repeat=True), axis=1)
		return df


	"""
	def pre_process_language(self, language='tamil'):
		lang_train_df = self._prepare_language_dataset_pandas(language=language, env=C.ENVIRONMENT_TRAIN)
		lang_validation_df = self._prepare_language_dataset_pandas(language=language, env=C.ENVIRONMENT_DEV)
		print('Describing the data frame category')
		print(lang_train_df.category.value_counts())
		df = lang_train_df[lang_train_df.category != 'not-Tamil']
		print('Describing the data frame category')
		print(df.category.value_counts())
		return None
	"""