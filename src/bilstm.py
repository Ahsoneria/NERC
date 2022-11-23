# from bilstm import bilstm_model
import operator
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras import layers
from keras import optimizers
from keras.models import Model
from keras_contrib.layers import CRF
from keras_contrib import losses
from keras_contrib import metrics
import string
import regex as re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class bilstm_model:
  def __init__(self):
    self.preprocessed_data = self.data_preprocessing()
    self.ner_model = self.train_BILSTMCRF(self.preprocessed_data)
    return

  def to_tuples(self, data):
      iterator = zip(data["Word"].values.tolist(),
                    data["POS"].values.tolist(),
                    data["Tag"].values.tolist())
      return [(word, pos, tag) for word, pos, tag in iterator]


  def data_preprocessing(self):
      data_df = pd.read_csv("buffer/nerc/src/ner_dataset.csv", encoding="iso-8859-1", header=0)

      data_df = data_df.fillna(method="ffill")
      data_df["Sentence #"] = data_df["Sentence #"].apply(lambda s: s[9:])
      data_df["Sentence #"] = data_df["Sentence #"].astype("int32")

      word_counts = data_df.groupby("Sentence #")["Word"].agg(["count"])
      word_counts = word_counts.rename(columns={"count": "Word count"})

      MAX_SENTENCE = word_counts.max()[0]

      all_words = list(set(data_df["Word"].values))
      all_tags = list(set(data_df["Tag"].values))

      word2index = {word: idx + 2 for idx, word in enumerate(all_words)}

      word2index["--UNKNOWN_WORD--"] = 0

      word2index["--PADDING--"] = 1

      index2word = {idx: word for word, idx in word2index.items()}

      for k, v in sorted(word2index.items(), key=operator.itemgetter(1))[:10]:
          print(k, v)

      tag2index = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
      tag2index["--PADDING--"] = 0

      index2tag = {idx: word for word, idx in tag2index.items()}

      sentences = data_df.groupby("Sentence #").apply(self.to_tuples).tolist()

      X = [[word[0] for word in sentence] for sentence in sentences]
      y = [[word[2] for word in sentence] for sentence in sentences]

      X = [[word2index[word] for word in sentence] for sentence in X]
      y = [[tag2index[tag] for tag in sentence] for sentence in y]

      X = [sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X]
      y = [sentence + [tag2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y]

      TAG_COUNT = len(tag2index)
      y = [np.eye(TAG_COUNT)[sentence] for sentence in y]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

      X_train = np.array(X_train)
      y_train = np.array(y_train)

      return [X_train, y_train, len(index2word), MAX_SENTENCE, TAG_COUNT, word2index, index2tag]


  def train_BILSTMCRF(self, preprocessed_data):
      X_train = preprocessed_data[0]
      y_train = preprocessed_data[1]
      WORD_COUNT = preprocessed_data[2]
      MAX_SENTENCE = preprocessed_data[3]
      TAG_COUNT = preprocessed_data[4]

      DENSE_EMBEDDING = 50
      LSTM_UNITS = 50
      LSTM_DROPOUT = 0.1
      DENSE_UNITS = 100
      BATCH_SIZE = 256
      MAX_EPOCHS = 5

      input_layer = layers.Input(shape=(MAX_SENTENCE,))

      model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", input_length=MAX_SENTENCE)(
          input_layer)
      model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)
      model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)

      crf_layer = CRF(units=TAG_COUNT)
      output_layer = crf_layer(model)

      ner_model = Model(input_layer, output_layer)

      loss = losses.crf_loss
      acc_metric = metrics.crf_accuracy
      opt = optimizers.Adam(lr=0.001)

      ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])
      ner_model.summary()

      history = ner_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1, verbose=2)

      return ner_model

  def test_model(self, sentence):
    MAX_SENTENCE = self.preprocessed_data[3]
    word2index = self.preprocessed_data[5]
    index2tag = self.preprocessed_data[6]

    re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
    sentence = re_tok.sub(r"  ", sentence).split()

    padded_sentence = sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence))
    padded_sentence = [word2index.get(w, 0) for w in padded_sentence]

    pred = self.ner_model.predict(np.array([padded_sentence]))
    pred = np.argmax(pred, axis=-1)

    res = ""
    for w, p in zip(sentence, pred[0]):
        res += "{:15}: {:5}".format(w, index2tag[p])

    # print(res)
    return res
