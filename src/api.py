# import nltk
import spacy
# from spacy import displacy
# from collections import Counter
# import en_core_web_sm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle

class NercAPI:
    def __init__(self):
      nltk.download('punkt')
      nltk.download('averaged_perceptron_tagger')
      nltk.download('universal_tagset')
      nltk.download('stopwords')
      self.spacy_nlp = spacy.load("en_core_web_sm")
      tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
      model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
      self.bert_ner = pipeline("ner", model=model, tokenizer=tokenizer)
      # self.bilstm_m = bilstm_model()
      return

    def preprocess_text(self, document):
        sentences = str(document.strip())
        return sentences

    def get_spacy_classification(self, preprocessed):
        doc = self.spacy_nlp(preprocessed)
        if not doc.ents:
          return [{}, {}]
        ent_list = [(X.start_char, X.text, X.label_) for X in doc.ents]
        entity_map = dict()
        color_map = dict()
        for ent in ent_list:
          color = ''
          if ('PER' in ent[2]):
            color = 'person'
          elif('GPE' in ent[2]):
            color = 'location'
          elif('ORG' in ent[2]):
            color = 'org'
          if color:
            entity_map[int(ent[0])] = len(ent[1])
            color_map[int(ent[0])] = color
        return [{k: v for k, v in sorted(entity_map.items(), key=lambda item: item[0])}, color_map]

    def get_bert_classification(self, preprocessed):
        """
        TODO 
        """
        ent_list = self.bert_ner(preprocessed)
        if not ent_list:
          return [{}, {}]
        entity_map = dict()
        color_map = dict()
        for ent in ent_list:
          color = ''
          if ('PER' in ent['entity']):
            color = 'person'
          elif('LOC' in ent['entity']):
            color = 'location'
          elif('ORG' in ent['entity']):
            color = 'org'
          if color:
            entity_map[int(ent['start'])] = int(ent['end'])-int(ent['start'])
            color_map[int(ent['start'])] = color

        return [{k: v for k, v in sorted(entity_map.items(), key=lambda item: item[0])}, color_map]

    def get_bilstm_classification(self, preprocessed):
        """
        TODO 
        """
        # result = self.bilstm_m(preprocessed)
        return ''

    def get_crf(self, preprocess):
      def pos_tags(document):
          sentences = nltk.sent_tokenize(document) 
          sentences = [nltk.word_tokenize(sent) for sent in sentences]
          sentences = [nltk.pos_tag(sent) for sent in sentences]
          return sentences

      with open('NERC/src/crf.pkl', "rb") as input_file:
        model = pickle.load(input_file)

      wordsList = pos_tags(preprocess)
      labels = model.predict(wordsList)
      if not labels or not labels[0]:
        return [{}, {}]
      print(wordsList, labels)
      start = 0
      entity_map = dict()
      color_map = dict()
      for i in range(len(labels[0])):
        color = ''
        if ('PER' in labels[0][i]):
          color = 'person'
        elif('LOC' in labels[0][i]):
          color = 'location'
        elif('ORG' in labels[0][i]):
          color = 'org'
        if color:
          entity_map[start] = len(wordsList[0][i][1])
          color_map[start] = color
        start += len(wordsList[0][i][1])+1
      return [entity_map, color_map]

    def get_baseline(self, preprocessed):
      start = 0
      entities_list = dict()
      wordsList = preprocessed.split(' ')
      for j in wordsList:
        tags = nltk.pos_tag([j])
        if len(tags)==1 and (tags[0][1] == 'NNP' or tags[0][1] == 'NNPS'):
          entities_list[start] = len(j)
        start += len(j)+1
      return entities_list

    def get_nerc_result(self, input_text):
        preprocessed = self.preprocess_text(input_text)
        output = []
        output.append(self.get_spacy_classification(preprocessed))
        output.append(self.get_bert_classification(preprocessed))
        output.append(self.get_bilstm_classification(preprocessed))
        output.append(self.get_crf(preprocessed))
        output.append(self.get_baseline(preprocessed))
        return output



