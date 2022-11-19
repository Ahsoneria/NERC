# import nltk
# import spacy
# from spacy import displacy
# from collections import Counter
# import en_core_web_sm


class NercAPI:
	def __init__(self):
		# self.nlp = spacy.load("en_core_web_sm")
		pass

	def preprocess_text(self, document):
		"""
		TODO 
		"""
		# sentences = document.split(' ')
		sentences = document.strip()
		return sentences

	def get_crf_classification(self):
		"""
		TODO 
		"""
		return

	def get_bert_classification(self):
		"""
		TODO 
		"""
		return

	def get_bilstm_classification(self):
		"""
		TODO 
		"""
		return

	def get_nerc_result(self, input_text):
		preprocessed = self.preprocess_text(input_text)
		# doc = self.nlp(preprocessed)
		# return [(X.text, X.label_) for X in doc.ents]
		return preprocessed
